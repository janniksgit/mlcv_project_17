
def visualize(rag, overseg, edge_values, image):
    shape = overseg.shape

    for x in range(shape[0]):
        for y in range(shape[1]):

            lu = overseg[x,y]

            if x + 1 < shape[0]:
                lv = overseg[x+1,y]

                if lu != lv :
                    e = rag.findEdge(lu, lv)

                    # normalization?
                    image[x,y]   = edge_values[e]
                    image[x+1,y] = edge_values[e]


            if y + 1 < shape[1]:

                lv = overseg[x,y+1]

                if lu != lv :
                    e = rag.findEdge(lu, lv)

                    # normalization?
                    image[x,y]   = edge_values[e]
                    image[x,y+1] = edge_values[e]

    return image


