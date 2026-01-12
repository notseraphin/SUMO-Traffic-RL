import sumolib

net = sumolib.net.readNet("sumo/network/grid.net.xml")
edges = [e.getID() for e in net.getEdges()]
print(edges)