def generate_mappings(registration):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node, node_name, display_name in registration:
        node_class_mappings[node_name] = node
        node_display_name_mappings[node_name] = display_name

    return node_class_mappings, node_display_name_mappings
