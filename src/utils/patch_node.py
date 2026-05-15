"""
PatchNode — hierarchy node for patch trees
==========================================

Represents a node in the WSI patch hierarchy, used for explicit tree
construction when a full patch tree is required.
"""

# ==========================================================================
# Imports
# ==========================================================================

from typing import Optional
from .wsi import WSI


# ==========================================================================
# Patch tree node
# ==========================================================================


class PatchNode:
    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        depth: int,
        max_depth: int,
        x: int,
        y: int,
        wsi: WSI,
    ):
        """
        Initialize a PatchNode.

        Args:
        -- depth (int): how many times this node has been zoomed in from the root
        -- max_depth (int): maximum allowed depth (zoom levels) for any node
        -- x (int): x coordinate of this patch in the WSI grid at its level
        -- y (int): y coordinate of this patch in the WSI grid at its level
        -- wsi (WSI): WSI object needed for expansion

        Returns:
            None: Description.
        """
        self.depth = depth
        self.max_depth = max_depth
        self.x = x
        self.y = y

        self.parent: Optional["PatchNode"] = None
        self.children: list["PatchNode"] = []

        self._expand_children(wsi)

    # ---------------------------------------------------------------------------
    # _expand_children
    # ---------------------------------------------------------------------------
    def _expand_children(self, wsi):
        """
        Fully initialize all children of this node (one level down),
        if depth < max_depth. Uses WSI geometry.

        Args:
            self: Description.
            wsi: Description.

        Returns:
            object: Description.
        """
        if not self.can_zoom():
            return

        # already expanded
        if self.children:
            return

        # parent level in WSI pyramid
        parent_level = wsi.max_level - self.depth
        child_level = parent_level - 1

        if child_level < wsi.min_level:
            return

        # get child grids (absolute coordinates)
        child_grids = wsi.get_child_grid(parent_level, self.x, self.y)
        if not child_grids:
            return

        # Iterate over child grids and create child nodes
        for grid in child_grids:
            # Iterate over child grid coordinates and create child nodes
            for cx, cy in grid:
                child_node = PatchNode(
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    x=cx,
                    y=cy,
                    wsi=wsi,
                )
                child_node.parent = self
                self.children.append(child_node)

    # ---------------------------------------------------------------------------
    # is_leaf
    # ---------------------------------------------------------------------------
    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf (i.e. has no children).

        Args:
            self: Description.

        Returns:
            bool: Description.
        """
        return len(self.children) == 0

    # ---------------------------------------------------------------------------
    # remaining_depth
    # ---------------------------------------------------------------------------
    def remaining_depth(self) -> int:
        """
        Return how many more times this node can be zoomed in before reaching max_depth.

        Args:
            self: Description.

        Returns:
            int: Description.
        """
        return self.max_depth - self.depth

    # ---------------------------------------------------------------------------
    # can_zoom
    # ---------------------------------------------------------------------------
    def can_zoom(self) -> bool:
        """
        Check if this node can be zoomed in (i.e. has children) based on depth.

        Args:
            self: Description.

        Returns:
            bool: Description.
        """
        return self.depth < self.max_depth

    # ---------------------------------------------------------------------------
    # list_all_descendants
    # ---------------------------------------------------------------------------
    def list_all_descendants(
        node: "PatchNode",
        wsi,
        include_self: bool = False,
    ) -> list["PatchNode"]:
        """
        Recursively list all descendants of `node` down to max_depth.

        Args:
        -- node (PatchNode): starting PatchNode (typically a root)
        -- wsi (WSI): WSI object needed for expansion
        -- include_self (bool): whether to include `node` itself

        Returns:
        -- List of PatchNode objects (all descendants)
        """
        result = []

        if include_self:
            result.append(node)

        if not node.can_zoom():
            return result

        children = node.expand(wsi)

        for child in children:
            result.append(child)
            result.extend(
                PatchNode.list_all_descendants(child, wsi, include_self=False)
            )

        return result
