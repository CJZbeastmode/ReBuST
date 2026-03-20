from typing import Optional
from .wsi import WSI

class PatchNode:
    def __init__(
        self,
        depth: int,
        max_depth: int,
        x: int,
        y: int,
        wsi: WSI,
    ):
        self.depth = depth
        self.max_depth = max_depth
        self.x = x
        self.y = y

        self.parent: Optional["PatchNode"] = None
        self.children: list["PatchNode"] = []

        self._expand_children(wsi)


    def _expand_children(self, wsi):
        """
        Fully initialize all children of this node (one level down),
        if depth < max_depth. Uses WSI geometry.

        This method is deterministic and idempotent:
        - calling it twice does nothing extra
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

        for grid in child_grids:
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


    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def remaining_depth(self) -> int:
        return self.max_depth - self.depth

    def can_zoom(self) -> bool:
        return self.depth < self.max_depth

    def list_all_descendants(
        node: "PatchNode",
        wsi,
        include_self: bool = False,
    ) -> list["PatchNode"]:
        """
        Recursively list all descendants of `node` down to max_depth.
        Children are expanded lazily using node.expand(wsi).

        Args:
            node: starting PatchNode (typically a root)
            wsi: WSI object needed for expansion
            include_self: whether to include `node` itself

        Returns:
            List of PatchNode objects (all descendants)
        """
        result = []

        if include_self:
            result.append(node)

        if not node.can_zoom():
            return result

        children = node.expand(wsi)

        for child in children:
            result.append(child)
            result.extend(PatchNode.list_all_descendants(child, wsi, include_self=False))

        return result
