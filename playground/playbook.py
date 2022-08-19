m = 10
edges = [(4, 9), (3, 9)]
# index range of incoming edges for each output node
edge_index_range = []
_start_pointer = _end_pointer = 0

for j in range(m):
    print(edge_index_range)
    while edges[_end_pointer][1] == j:
        _end_pointer += 1
        if _end_pointer == len(edges):
            break
    edge_index_range.append((_start_pointer, _end_pointer))
    _start_pointer = _end_pointer

print(edge_index_range)
