import copy
import datetime
import glob
import logging
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Set, Tuple, cast

import dateutil.parser
import networkx as nx
import orjson
import requests
import tqdm
import yaml
from conda.models.version import VersionOrder
from graphviz import Source

logger = logging.getLogger(__name__)

from conda_forge_tick.contexts import FeedstockContext, MigratorSessionContext
from conda_forge_tick.lazy_json_backends import LazyJson, get_all_keys_for_hashmap
from conda_forge_tick.make_migrators import load_migrators
from conda_forge_tick.migrators import (
    ArchRebuild,
    GraphMigrator,
    MatplotlibBase,
    MigrationYamlCreator,
    Migrator,
    OSXArm,
    Replacement,
    Version,
    WinArm64,
)
from conda_forge_tick.os_utils import eval_cmd
from conda_forge_tick.path_lengths import cyclic_topological_sort
from conda_forge_tick.utils import (
    fold_log_lines,
    frozen_to_json_friendly,
    load_existing_graph,
    pr_can_be_archived,
)
from conda_forge_tick.version_filters import filter_version

GH_MERGE_STATE_STATUS = [
    "behind",
    "blocked",
    "clean",
    "dirty",
    "draft",
    "has_hooks",
    "unknown",
    "unstable",
]


def _sorted_set_json(obj: Any) -> Any:
    """If obj is a set, return sorted(obj). Else, raise TypeError.

    Used for custom object serialization.

    Raises
    ------
    TypeError
        If obj is not a set.
    """
    if isinstance(obj, Set):
        return sorted(obj)
    raise TypeError(repr(obj) + " is not JSON serializable")


def _ok_version(ver):
    return ver is not None and ver and isinstance(ver, str)


def write_version_migrator_status(migrator, mctx):
    """Write the status of the version migrator."""
    out: Dict[str, Dict[str, str]] = {
        "queued": {},  # name -> pending version
        "errors": {},  # name -> error
    }

    gx = mctx.graph
    version_nodes = get_all_keys_for_hashmap("versions")

    for node in version_nodes:
        version_data = LazyJson(f"versions/{node}.json").data
        with gx.nodes[f"{node}"]["payload"] as attrs:
            if attrs.get("archived", False):
                continue

            with attrs["version_pr_info"] as vpri:
                version_from_data = filter_version(
                    attrs,
                    version_data.get("new_version", False),
                )
                version_from_attrs = filter_version(
                    attrs,
                    vpri.get("new_version", False),
                )
                if _ok_version(version_from_data):
                    if _ok_version(version_from_attrs):
                        new_version: str | bool = max(
                            [version_from_data, version_from_attrs],
                            key=lambda x: VersionOrder(x.replace("-", ".")),  # type: ignore[union-attr]
                        )
                    else:
                        new_version = version_from_data
                else:
                    new_version = vpri.get("new_version", False)

                try:
                    if "new_version" in vpri:
                        old_vpri_version = vpri["new_version"]
                        had_vpri_version = True
                    else:
                        had_vpri_version = False

                    vpri["new_version"] = new_version

                    new_version_is_ok = _ok_version(
                        new_version
                    ) and not migrator.filter(attrs)
                finally:
                    if had_vpri_version:
                        vpri["new_version"] = old_vpri_version
                    else:
                        del vpri["new_version"]

                # run filter with new_version
                if new_version_is_ok:
                    attempts = vpri.get("new_version_attempts", {}).get(new_version, 0)
                    if attempts == 0:
                        out["queued"][node] = new_version  # type: ignore[assignment]
                    else:
                        out["errors"][node] = f"{attempts:.2f} attempts - " + vpri.get(
                            "new_version_errors",
                            {},
                        ).get(
                            new_version,
                            "No error information available for version '%s'."
                            % new_version,
                        )

    with open("./status/version_status.json", "wb") as f:
        old_out: dict[str, dict[str, str] | set[str]] = {}
        old_out.update(out)
        old_out["queued"] = set(out["queued"].keys())
        old_out["errored"] = set(out["errors"].keys())
        f.write(
            orjson.dumps(
                old_out,
                option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
                default=_sorted_set_json,
            )
        )

    with open("./status/version_status.v2.json", "wb") as f:
        f.write(
            orjson.dumps(
                out,
                option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
                default=_sorted_set_json,
            )
        )


def graph_migrator_status(
    migrator: Migrator,
    gx: nx.DiGraph,
) -> Tuple[dict, list, nx.DiGraph]:
    """Get the migrator progress for a given migrator."""
    migrator_name = migrator.report_name

    num_viz = 0

    out: Dict[str, Set[str]] = {
        "done": set(),
        "in-pr": set(),
        "awaiting-pr": set(),
        "not-solvable": set(),
        "awaiting-parents": set(),
        "bot-error": set(),
    }

    gx2 = copy.deepcopy(getattr(migrator, "graph", gx))

    top_level = {node for node in gx2 if not list(gx2.predecessors(node))}
    build_sequence = list(cyclic_topological_sort(gx2, top_level))

    feedstock_metadata = dict()

    import graphviz
    from streamz.graph import _clean_text

    gv = graphviz.Digraph(graph_attr={"packmode": "array_3"})

    # pinning isn't actually in the migration
    if "conda-forge-pinning" in gx2.nodes():
        gx2.remove_node("conda-forge-pinning")

    # Check if pyside2 and matplotlib are in the graph
    all_nodes = list(gx2.nodes.keys())
    pyside2_nodes_in_graph = []
    matplotlib_nodes_in_graph = []
    for n in all_nodes:
        if n == "pyside2" or n == "pyside2-feedstock":
            pyside2_nodes_in_graph.append(n)
        elif n == "matplotlib" or n == "matplotlib-feedstock":
            matplotlib_nodes_in_graph.append(n)
        else:
            try:
                with gx2.nodes[n]["payload"] as attrs:
                    name = attrs.get("name", "")
                    feedstock_name = attrs.get("feedstock_name", "")
                    if (name == "pyside2" or feedstock_name == "pyside2" or 
                        feedstock_name == "pyside2-feedstock"):
                        pyside2_nodes_in_graph.append(n)
                    elif (name == "matplotlib" or feedstock_name == "matplotlib" or 
                          feedstock_name == "matplotlib-feedstock"):
                        matplotlib_nodes_in_graph.append(n)
            except Exception:
                pass
    
    print(f"[DEBUG status_report] Total nodes to process: {len(all_nodes)}", flush=True)
    if pyside2_nodes_in_graph:
        print(f"[DEBUG pyside2 status_report] pyside2 found in graph: {pyside2_nodes_in_graph}", flush=True)
        # Check matplotlib relationship to pyside2
        for pyside2_node in pyside2_nodes_in_graph:
            if pyside2_node in gx2.nodes:
                try:
                    predecessors = list(gx2.predecessors(pyside2_node))
                    successors = list(gx2.successors(pyside2_node))
                    matplotlib_predecessors = [p for p in predecessors if p == "matplotlib" or p == "matplotlib-feedstock" or 
                                              (p in gx2.nodes and 
                                               gx2.nodes[p].get("payload", {}).get("name") == "matplotlib")]
                    matplotlib_successors = [s for s in successors if s == "matplotlib" or s == "matplotlib-feedstock" or 
                                            (s in gx2.nodes and 
                                             gx2.nodes[s].get("payload", {}).get("name") == "matplotlib")]
                    print(f"[DEBUG pyside2 status_report] pyside2 node {pyside2_node} has {len(predecessors)} predecessors, {len(successors)} successors", flush=True)
                    if matplotlib_predecessors:
                        print(f"[DEBUG pyside2 status_report] matplotlib is a PREDECESSOR (dependency) of pyside2: {matplotlib_predecessors}", flush=True)
                    if matplotlib_successors:
                        print(f"[DEBUG pyside2 status_report] matplotlib is a SUCCESSOR (dependent) of pyside2: {matplotlib_successors}", flush=True)
                    # Check all predecessors for matplotlib
                    for pred in predecessors[:10]:  # Check first 10
                        try:
                            with gx2.nodes[pred]["payload"] as pred_attrs:
                                pred_name = pred_attrs.get("name", "")
                                if "matplotlib" in pred_name.lower():
                                    print(f"[DEBUG pyside2 status_report] pyside2 depends on: {pred} (name={pred_name})", flush=True)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[DEBUG pyside2 status_report] Error checking relationships: {e}", flush=True)
    else:
        print(f"[DEBUG pyside2 status_report] pyside2 NOT found in graph nodes", flush=True)
        # Check all nodes for pyside2
        pyside_related = []
        for check_node in all_nodes:
            try:
                with gx2.nodes[check_node]["payload"] as check_attrs:
                    check_name = check_attrs.get("name", "")
                    check_feedstock = check_attrs.get("feedstock_name", "")
                    if "pyside" in check_name.lower() or "pyside" in check_feedstock.lower() or "pyside" in check_node.lower():
                        pyside_related.append((check_node, check_name, check_feedstock))
            except Exception:
                pass
        if pyside_related:
            print(f"[DEBUG status_report] Found {len(pyside_related)} pyside-related nodes:", flush=True)
            for node, name, feedstock in pyside_related[:20]:  # Show first 20
                print(f"[DEBUG status_report]   - {node} (name={name}, feedstock={feedstock})", flush=True)
    
    if matplotlib_nodes_in_graph:
        print(f"[DEBUG matplotlib status_report] matplotlib found in graph: {matplotlib_nodes_in_graph}", flush=True)
    else:
        print(f"[DEBUG matplotlib status_report] matplotlib NOT found in graph nodes", flush=True)

    for idx, (node, node_attrs) in enumerate(gx2.nodes.items()):
        attrs = node_attrs["payload"]
        node_name = attrs.get("name", "")
        feedstock_name = attrs.get("feedstock_name", "")
        # Check if this is pyside2: node name, package name, or feedstock name
        is_pyside2 = (node == "pyside2" or node == "pyside2-feedstock" or 
                     node_name == "pyside2" or feedstock_name == "pyside2" or 
                     feedstock_name == "pyside2-feedstock")
        # Check if this is matplotlib: node name, package name, or feedstock name
        is_matplotlib = (node == "matplotlib" or node == "matplotlib-feedstock" or 
                        node_name == "matplotlib" or feedstock_name == "matplotlib" or 
                        feedstock_name == "matplotlib-feedstock")
        
        # Print progress every 100 packages or for pyside2/matplotlib
        if is_pyside2 or is_matplotlib or idx % 100 == 0:
            print(f"[DEBUG status_report] Processing package {idx+1}/{len(all_nodes)}: node={node}, name={node_name}, feedstock={feedstock_name}", flush=True)
        
        if is_pyside2:
            print(f"[DEBUG pyside2 status_report] ===== PROCESSING NODE: node={node}, name={node_name}, feedstock_name={feedstock_name} =====", flush=True)
            print(f"[DEBUG pyside2 status_report] archived={attrs.get('archived', False)}", flush=True)
        
        if is_matplotlib:
            print(f"[DEBUG matplotlib status_report] ===== PROCESSING NODE: node={node}, name={node_name}, feedstock_name={feedstock_name} =====", flush=True)
            print(f"[DEBUG matplotlib status_report] archived={attrs.get('archived', False)}", flush=True)
            # Check relationship to pyside2
            try:
                predecessors = list(gx2.predecessors(node))
                successors = list(gx2.successors(node))
                pyside2_predecessors = [p for p in predecessors if p == "pyside2" or p == "pyside2-feedstock" or 
                                       (p in gx2.nodes and 
                                        gx2.nodes[p].get("payload", {}).get("name") == "pyside2")]
                pyside2_successors = [s for s in successors if s == "pyside2" or s == "pyside2-feedstock" or 
                                     (s in gx2.nodes and 
                                      gx2.nodes[s].get("payload", {}).get("name") == "pyside2")]
                if pyside2_predecessors:
                    print(f"[DEBUG matplotlib status_report] pyside2 is a PREDECESSOR (dependency) of matplotlib: {pyside2_predecessors}", flush=True)
                if pyside2_successors:
                    print(f"[DEBUG matplotlib status_report] pyside2 is a SUCCESSOR (dependent) of matplotlib: {pyside2_successors}", flush=True)
            except Exception as e:
                print(f"[DEBUG matplotlib status_report] Error checking pyside2 relationship: {e}", flush=True)
        
        # remove archived from status
        if attrs.get("archived", False):
            if is_pyside2:
                print(f"[DEBUG pyside2 status_report] SKIPPING (archived)", flush=True)
            if is_matplotlib:
                print(f"[DEBUG matplotlib status_report] SKIPPING (archived)", flush=True)
            continue
        node_metadata: Dict = {}
        feedstock_metadata[node] = node_metadata
        nuid = migrator.migrator_uid(attrs)
        all_pr_jsons = []
        for pr_json in attrs.get("pr_info", {}).get("PRed", []):
            all_pr_jsons.append(copy.deepcopy(pr_json))

        feedstock_ctx = FeedstockContext(
            feedstock_name=attrs.get("feedstock_name", node),
            attrs=attrs,
        )

        # hack around bug in migrator vs graph data for this one
        if isinstance(migrator, MatplotlibBase):
            if "name" in nuid:
                del nuid["name"]
            for i in range(len(all_pr_jsons)):
                if (
                    all_pr_jsons[i]
                    and "name" in all_pr_jsons[i]["data"]
                    and all_pr_jsons[i]["data"]["migrator_name"] == "MatplotlibBase"
                ):
                    del all_pr_jsons[i]["data"]["name"]

        for pr_json in all_pr_jsons:
            if pr_json and pr_json["data"] == frozen_to_json_friendly(nuid)["data"]:
                break
        else:
            pr_json = None

        # No PR was ever issued but the migration was performed.
        # This is only the case when the migration was done manually
        # before the bot could issue any PR.
        manually_done = pr_json is None and frozen_to_json_friendly(nuid)["data"] in (
            z["data"] for z in all_pr_jsons
        )

        if pr_json is not None and "PR" in pr_json:
            pr_is_archiveable = pr_can_be_archived(pr_json["PR"])
        else:
            pr_is_archiveable = False

        node_name = attrs.get("name", "")
        feedstock_name = attrs.get("feedstock_name", "")
        # Check if this is pyside2: node name, package name, or feedstock name
        is_pyside2 = (node == "pyside2" or node == "pyside2-feedstock" or 
                     node_name == "pyside2" or feedstock_name == "pyside2" or 
                     feedstock_name == "pyside2-feedstock")
        # Check if this is matplotlib: node name, package name, or feedstock name
        is_matplotlib = (node == "matplotlib" or node == "matplotlib-feedstock" or 
                        node_name == "matplotlib" or feedstock_name == "matplotlib" or 
                        feedstock_name == "matplotlib-feedstock")
        
        if is_pyside2:
            print(f"[DEBUG pyside2 status_report] START: node={node}, name={node_name}, feedstock_name={feedstock_name}", flush=True)
            print(f"[DEBUG pyside2 status_report] migrator_name={migrator_name}", flush=True)
            print(f"[DEBUG pyside2 status_report] migrator_class={migrator.__class__.__name__}", flush=True)
            print(f"[DEBUG pyside2 status_report] calling migrator.filter(attrs)...", flush=True)
        
        if is_matplotlib:
            print(f"[DEBUG matplotlib status_report] START: node={node}, name={node_name}, feedstock_name={feedstock_name}", flush=True)
            print(f"[DEBUG matplotlib status_report] migrator_name={migrator_name}", flush=True)
            print(f"[DEBUG matplotlib status_report] migrator_class={migrator.__class__.__name__}", flush=True)
            print(f"[DEBUG matplotlib status_report] calling migrator.filter(attrs)...", flush=True)
        
        filter_result = migrator.filter(attrs)
        buildable = not filter_result
        
        if is_pyside2:
            print(f"[DEBUG pyside2 status_report] migrator.filter() returned={filter_result}", flush=True)
            print(f"[DEBUG pyside2 status_report] buildable={buildable}", flush=True)
            print(f"[DEBUG pyside2 status_report] pr_json is None={pr_json is None}", flush=True)
            print(f"[DEBUG pyside2 status_report] manually_done={manually_done}", flush=True)
            print(f"[DEBUG pyside2 status_report] pr_is_archiveable={pr_is_archiveable}", flush=True)
            pre_pr_status = attrs.get("pr_info", {}).get("pre_pr_migrator_status", {}).get(migrator_name, "")
            print(f"[DEBUG pyside2 status_report] pre_pr_migrator_status={pre_pr_status}", flush=True)
            print(f"[DEBUG pyside2 status_report] parsing_error={attrs.get('parsing_error', 'N/A')}", flush=True)
        
        if is_matplotlib:
            print(f"[DEBUG matplotlib status_report] migrator.filter() returned={filter_result}", flush=True)
            print(f"[DEBUG matplotlib status_report] buildable={buildable}", flush=True)
            print(f"[DEBUG matplotlib status_report] pr_json is None={pr_json is None}", flush=True)
            print(f"[DEBUG matplotlib status_report] manually_done={manually_done}", flush=True)
            print(f"[DEBUG matplotlib status_report] pr_is_archiveable={pr_is_archiveable}", flush=True)
            pre_pr_status = attrs.get("pr_info", {}).get("pre_pr_migrator_status", {}).get(migrator_name, "")
            print(f"[DEBUG matplotlib status_report] pre_pr_migrator_status={pre_pr_status}", flush=True)
            print(f"[DEBUG matplotlib status_report] parsing_error={attrs.get('parsing_error', 'N/A')}", flush=True)
        
        fntc = "black"
        status_icon = ""
        if manually_done:
            out["done"].add(node)
            fc = "#440154"
            fntc = "white"
            if is_pyside2:
                print(f"[DEBUG pyside2 status_report] STATUS: done (manually_done)", flush=True)
            if is_matplotlib:
                print(f"[DEBUG matplotlib status_report] STATUS: done (manually_done)", flush=True)
        elif pr_json is None:
            if buildable:
                if "not solvable" in (
                    attrs.get("pr_info", {})
                    .get("pre_pr_migrator_status", {})
                    .get(migrator_name, "")
                ):
                    out["not-solvable"].add(node)
                    fc = "#ff8c00"
                    if is_pyside2:
                        print(f"[DEBUG pyside2 status_report] STATUS: not-solvable", flush=True)
                    if is_matplotlib:
                        print(f"[DEBUG matplotlib status_report] STATUS: not-solvable", flush=True)
                elif "bot error" in (
                    attrs.get("pr_info", {})
                    .get("pre_pr_migrator_status", {})
                    .get(migrator_name, "")
                ) or attrs.get("parsing_error", ""):
                    out["bot-error"].add(node)
                    fc = "#000000"
                    fntc = "white"
                    if is_pyside2:
                        print(f"[DEBUG pyside2 status_report] STATUS: bot-error", flush=True)
                    if is_matplotlib:
                        print(f"[DEBUG matplotlib status_report] STATUS: bot-error", flush=True)
                else:
                    out["awaiting-pr"].add(node)
                    fc = "#35b779"
                    if is_pyside2:
                        print(f"[DEBUG pyside2 status_report] STATUS: awaiting-pr", flush=True)
                    if is_matplotlib:
                        print(f"[DEBUG matplotlib status_report] STATUS: awaiting-pr", flush=True)
            else:
                if "bot error" in (
                    attrs.get("pr_info", {})
                    .get("pre_pr_migrator_status", {})
                    .get(migrator_name, "")
                ) or attrs.get("parsing_error", ""):
                    out["bot-error"].add(node)
                    fc = "#000000"
                    fntc = "white"
                    if is_pyside2:
                        print(f"[DEBUG pyside2 status_report] STATUS: bot-error", flush=True)
                    if is_matplotlib:
                        print(f"[DEBUG matplotlib status_report] STATUS: bot-error", flush=True)
                else:
                    out["awaiting-parents"].add(node)
                    fc = "#fde725"
                    if is_pyside2:
                        print(f"[DEBUG pyside2 status_report] STATUS: awaiting-parents (buildable=False)", flush=True)
                    if is_matplotlib:
                        print(f"[DEBUG matplotlib status_report] STATUS: awaiting-parents (buildable=False)", flush=True)
        elif "PR" not in pr_json or "state" not in pr_json["PR"]:
            out["bot-error"].add(node)
            fc = "#000000"
            fntc = "white"
        elif pr_is_archiveable:
            out["done"].add(node)
            fc = "#440154"
            fntc = "white"
        else:
            out["in-pr"].add(node)
            fc = "#31688e"
            fntc = "white"
            pr_status = pr_json["PR"]["mergeable_state"]
            if pr_status == "clean":
                status_icon = " ✓"
            else:
                status_icon = " ❎"
        if node not in out["done"]:
            num_viz += 1
            gv.node(
                node,
                label=_clean_text(node) + status_icon,
                fillcolor=fc,
                style="filled",
                fontcolor=fntc,
                URL=(pr_json or {})
                .get("PR", {})
                .get(
                    "html_url",
                    feedstock_ctx.git_http_ref,
                ),
            )

        # additional metadata for reporting
        node_metadata["num_descendants"] = len(nx.descendants(gx2, node))
        node_metadata["immediate_children"] = [
            k
            for k in sorted(gx2.successors(node))
            if not gx2[k].get("payload", {}).get("archived", False)
        ]
        if node in out["not-solvable"] or node in out["bot-error"]:
            node_metadata["pre_pr_migrator_status"] = (
                attrs.get("pr_info", {})
                .get(
                    "pre_pr_migrator_status",
                    {},
                )
                .get(migrator_name, "")
            ) or attrs.get("parsing_error", "")
        else:
            node_metadata["pre_pr_migrator_status"] = ""

        if pr_json and "PR" in pr_json:
            # I needed to fake some PRs they don't have html_urls though
            node_metadata["pr_url"] = pr_json["PR"].get(
                "html_url",
                feedstock_ctx.git_http_ref,
            )
            node_metadata["pr_status"] = pr_json["PR"].get("mergeable_state", "")

            for timestamp_field in ["created_at", "updated_at"]:
                timestamp = pr_json["PR"].get(timestamp_field, None)
                if timestamp is not None:
                    timestamp = dateutil.parser.parse(timestamp)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
                    node_metadata[timestamp_field] = timestamp.isoformat()

    out2: Dict = {}
    for k in out.keys():
        out2[k] = list(
            sorted(
                out[k],
                key=lambda x: (
                    build_sequence.index(x) if x in build_sequence else -1,
                    x,
                ),
            ),
        )

    out2["_feedstock_status"] = feedstock_metadata
    for (e0, e1), edge_attrs in gx2.edges.items():
        if (
            e0 not in out["done"]
            and e1 not in out["done"]
            and not gx2.nodes[e0]["payload"].get("archived", False)
            and not gx2.nodes[e1]["payload"].get("archived", False)
        ):
            gv.edge(e0, e1)

    print("    len(gv):", num_viz, flush=True)
    out2["_num_viz"] = num_viz

    return out2, build_sequence, gv


def _collect_items_from_nodes(gx, func):
    futs = []
    with ThreadPoolExecutor(max_workers=20) as exec:
        for k in gx.nodes:
            futs.append(exec.submit(func, k))
        return [
            fut.result()
            for fut in tqdm.tqdm(as_completed(futs), total=len(futs), ncols=80)
            if fut.result() is not None
        ]


def _compute_recently_closed(total_status, old_closed_status, old_total_status):
    now = int(time.time())
    two_weeks = 14 * 24 * 60 * 60

    # grab any new stuff
    closed_status = {m: now for m in set(old_total_status) - set(total_status)}

    # grab anything recent from previous stuff
    for m, nm in old_closed_status.items():
        tm = int(dateutil.parser.parse(nm.split(" closed at ", 1)[1]).timestamp())
        if m not in total_status and now - tm < two_weeks:
            closed_status[m] = tm

    # now make it pretty
    closed_status = {
        k: (
            k
            + " closed at "
            + datetime.datetime.fromtimestamp(v).isoformat().replace("T", " ")
            + " UTC"
        )
        for k, v in closed_status.items()
    }

    return closed_status


def main(migrator_filter: str | None = None) -> None:
    with fold_log_lines("loading existing status data, graph and migrators"):
        r = requests.get(
            "https://raw.githubusercontent.com/conda-forge/"
            "conda-forge.github.io/77bb24125496/sphinx/img/anvil.svg",
        )

        # cache these for later
        if os.path.exists("status/closed_status.json"):
            with open("status/closed_status.json", "rb") as fp:
                old_closed_status = orjson.loads(fp.read())
        else:
            old_closed_status = {}

        with open("status/total_status.json", "rb") as fp:
            old_total_status = orjson.loads(fp.read())

        smithy_version: str = eval_cmd(["conda", "smithy", "--version"]).strip()
        pinning_version: str = cast(
            str,
            orjson.loads(eval_cmd(["conda", "list", "conda-forge-pinning", "--json"]))[
                0
            ]["version"],
        )
        # Use cached graph reference (read-only) to avoid deep copy
        print(f"[DEBUG status_report] loading graph...", flush=True)
        gx = load_existing_graph(deep_copy=False)
        print(f"[DEBUG status_report] loaded graph with {len(gx.nodes)} nodes, id={id(gx)}", flush=True)
        mctx = MigratorSessionContext(
            graph=gx,
            smithy_version=smithy_version,
            pinning_version=pinning_version,
        )
        print(f"[DEBUG status_report] loading migrators...", flush=True)
        migrators = load_migrators(skip_paused=False, filter_name=migrator_filter)
        print(f"[DEBUG status_report] loaded {len(migrators)} migrators", flush=True)

    os.makedirs("./status/migration_json", exist_ok=True)
    os.makedirs("./status/migration_svg", exist_ok=True)
    regular_status = {}
    longterm_status = {}
    paused_status = {}

    for migrator in migrators:
        # we do not show these on the status page since they are used to
        # open and close migrations
        if isinstance(migrator, MigrationYamlCreator):
            continue

        migrator_name = migrator.report_name

        print(
            "================================================================",
            flush=True,
        )
        print("name:", migrator_name, flush=True)
        print(f"[DEBUG status_report] Processing migrator: {migrator_name} ({migrator.__class__.__name__})", flush=True)
        
        # Check if this migrator's graph contains pyside2
        migrator_graph = getattr(migrator, "graph", None)
        if migrator_graph:
            migrator_nodes = list(migrator_graph.nodes.keys())
            pyside2_in_migrator_graph = []
            matplotlib_in_migrator_graph = []
            for n in migrator_nodes:
                if n == "pyside2" or n == "pyside2-feedstock":
                    pyside2_in_migrator_graph.append(n)
                elif n == "matplotlib" or n == "matplotlib-feedstock":
                    matplotlib_in_migrator_graph.append(n)
                else:
                    try:
                        with migrator_graph.nodes[n]["payload"] as attrs:
                            name = attrs.get("name", "")
                            feedstock_name = attrs.get("feedstock_name", "")
                            if (name == "pyside2" or feedstock_name == "pyside2" or 
                                feedstock_name == "pyside2-feedstock"):
                                pyside2_in_migrator_graph.append(n)
                            elif (name == "matplotlib" or feedstock_name == "matplotlib" or 
                                  feedstock_name == "matplotlib-feedstock"):
                                matplotlib_in_migrator_graph.append(n)
                    except Exception:
                        pass
            if pyside2_in_migrator_graph:
                print(f"[DEBUG pyside2 status_report] pyside2 found in migrator {migrator_name} graph: {pyside2_in_migrator_graph}", flush=True)
            else:
                print(f"[DEBUG pyside2 status_report] pyside2 NOT in migrator {migrator_name} graph (has {len(migrator_nodes)} nodes)", flush=True)
            if matplotlib_in_migrator_graph:
                print(f"[DEBUG matplotlib status_report] matplotlib found in migrator {migrator_name} graph: {matplotlib_in_migrator_graph}", flush=True)
            else:
                print(f"[DEBUG matplotlib status_report] matplotlib NOT in migrator {migrator_name} graph (has {len(migrator_nodes)} nodes)", flush=True)
        else:
            print(f"[DEBUG pyside2 status_report] migrator {migrator_name} has no graph attribute", flush=True)

        if (
            isinstance(migrator, GraphMigrator)
            or isinstance(migrator, Replacement)
            or isinstance(migrator, Migrator)
        ) and not isinstance(migrator, Version):
            if isinstance(migrator, GraphMigrator):
                mgconf = yaml.safe_load(getattr(migrator, "yaml_contents", "{}")).get(
                    "__migrator",
                    {},
                )
                if mgconf.get("paused", False):
                    paused_status[migrator_name] = f"{migrator.name} Migration Status"
                elif (
                    mgconf.get("longterm", False)
                    or isinstance(migrator, ArchRebuild)
                    or isinstance(migrator, OSXArm)
                    or isinstance(migrator, WinArm64)
                ):
                    longterm_status[migrator_name] = f"{migrator.name} Migration Status"
                else:
                    regular_status[migrator_name] = f"{migrator.name} Migration Status"
            else:
                regular_status[migrator_name] = f"{migrator.name} Migration Status"

            status, _, gv = graph_migrator_status(migrator, mctx.graph)
            num_viz = status.pop("_num_viz", 0)
            with open(
                os.path.join(f"./status/migration_json/{migrator_name}.json"),
                "wb",
            ) as fp:
                fp.write(
                    orjson.dumps(
                        status,
                        option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
                        default=_sorted_set_json,
                    )
                )

            if num_viz <= 500:
                d = gv.pipe("dot")
                with tempfile.NamedTemporaryFile(suffix=".dot") as ntf:
                    ntf.write(d)
                    # make the graph a bit more compact
                    d = Source(
                        subprocess.check_output(
                            ["unflatten", "-f", "-l", "5", "-c", "10", f"{ntf.name}"],
                        ).decode("utf-8"),
                    ).pipe("svg")
                with open(
                    os.path.join(f"./status/migration_svg/{migrator_name}.svg"),
                    "wb",
                ) as fb:
                    fb.write(d or gv.pipe("svg"))
            else:
                with open(
                    os.path.join(f"./status/migration_svg/{migrator_name}.svg"),
                    "wb",
                ) as fb:
                    fb.write(r.content)

        elif isinstance(migrator, Version):
            write_version_migrator_status(migrator, mctx)

        print(" ", flush=True)

    print("writing data", flush=True)
    with open("./status/regular_status.json", "wb") as f:
        f.write(
            orjson.dumps(
                regular_status,
                option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
                default=_sorted_set_json,
            )
        )

    with open("./status/longterm_status.json", "wb") as f:
        f.write(
            orjson.dumps(
                longterm_status,
                option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
                default=_sorted_set_json,
            )
        )

    with open("./status/paused_status.json", "wb") as f:
        f.write(
            orjson.dumps(
                paused_status,
                option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
                default=_sorted_set_json,
            )
        )

    total_status = {}
    total_status.update(regular_status)
    total_status.update(longterm_status)
    total_status.update(paused_status)
    with open("./status/total_status.json", "wb") as f:
        f.write(
            orjson.dumps(
                total_status,
                option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
                default=_sorted_set_json,
            )
        )

    closed_status = _compute_recently_closed(
        total_status,
        old_closed_status,
        old_total_status,
    )
    with open("./status/closed_status.json", "wb") as f:
        f.write(
            orjson.dumps(
                closed_status,
                option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2,
                default=_sorted_set_json,
            )
        )

    # remove old status files
    old_files = glob.glob("./status/migration_*/*.*")
    for old_file in old_files:
        mname = os.path.basename(old_file).rsplit(".", 1)[0]
        if (mname not in total_status) and (mname not in closed_status):
            subprocess.run(
                ["git", "rm", "-f", old_file],
                check=True,
            )

    # I have turned this off since we do not use it
    # MRB - 2023/03/08
    """
    print("\ncomputing feedstock and PR stats", flush=True)

    def _get_needs_help(k):
        v = mctx.graph.nodes[k]
        if (
            len(
                [
                    z
                    for z in v.get("payload", {}).get("pr_info", {}).get("PRed", [])
                    if z.get("PR", {}).get("state", "closed") == "open"
                    and z.get("data", {}).get("migrator_name", "") == Version.name
                ],
            )
            >= Version.max_num_prs
        ):
            return k
        else:
            return None

    lst = _collect_items_from_nodes(mctx.graph, _get_needs_help)
    with open("./status/could_use_help.json", "w") as f:
        json.dump(
            sorted(
                lst,
                key=lambda z: (len(nx.descendants(mctx.graph, z)), lst),
                reverse=True,
            ),
            f,
            indent=2,
        )

    lm = LicenseMigrator()

    def _get_needs_license(k):
        v = mctx.graph.nodes[k]
        if not lm.filter(v.get("payload", {})):
            return k
        else:
            return None

    lst = _collect_items_from_nodes(mctx.graph, _get_needs_license)
    with open("./status/unlicensed.json", "w") as f:
        json.dump(
            sorted(
                lst,
                key=lambda z: (len(nx.descendants(mctx.graph, z)), lst),
                reverse=True,
            ),
            f,
            indent=2,
        )

    lst = [
        k
        for k, v in mctx.graph.nodes.items()
        if v.get("payload", {}).get("archived", False)
    ]
    with open("./status/archived.json", "w") as f:
        json.dump(sorted(lst), f, indent=2)

    def _get_open_pr_states(k):
        attrs = mctx.graph.nodes[k]["payload"]
        _open_prs = []
        for pr in attrs.get("pr_info", {}).get("PRed", []):
            if pr.get("PR", {}).get("state", "closed") != "closed":
                _open_prs.append(pr["PR"])

        return _open_prs

    open_prs = []
    for op in _collect_items_from_nodes(mctx.graph, _get_open_pr_states):
        open_prs.extend(op)
    merge_state_count = Counter([o["mergeable_state"] for o in open_prs])
    with open("./status/pr_state.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([merge_state_count[k] for k in GH_MERGE_STATE_STATUS])
    """
