def print_status(behavior):
    print(f"{behavior.name}: {behavior.status}")

def wait_until_success(tree, max_ticks=1000):
    for i in range(max_ticks):
        tree.tick()
        if tree.root.status == py_trees.common.Status.SUCCESS:
            return True
    return False
