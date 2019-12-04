import tensorflow as tf


def construct_graph(graph_dict, inputs, outputs):
    queue = inputs[:]
    make_dict = {}
    for key, val in graph_dict.items():
        if key in inputs:
            # Use keras.Input instead of placeholders
            make_dict[key] = tf.keras.Input(name=key, shape=(), dtype=tf.dtypes.float32)
        else:
            make_dict[key] = None
    # Breadth-First search of graph starting from inputs
    while len(queue) != 0:
        cur = graph_dict[queue[0]]
        for outg in cur["outgoing"]:
            if make_dict[outg[0]] is not None: # If discovered node, do add/multiply operation
                make_dict[outg[0]] = tf.keras.layers.add([
                    make_dict[outg[0]],
                    tf.keras.layers.multiply(
                        [[outg[1]], make_dict[queue[0]]],
                    )],
                )
            else: # If undiscovered node, input is just coming in multiplied and add outgoing to queue
                make_dict[outg[0]] = tf.keras.layers.multiply(
                    [make_dict[queue[0]], [outg[1]]]
                )
                for outgo in graph_dict[outg[0]]["outgoing"]:
                    queue.append(outgo[0])
        queue.pop(0)
    # Returns one data graph for each output
    model_inputs = [make_dict[key] for key in inputs]
    model_outputs = [make_dict[key] for key in outputs]
    return [tf.keras.Model(inputs=model_inputs, outputs=o) for o in model_outputs]


def construct_network(genome):

    link_dict = {}

    for gene in genome.genes:
        if gene.output in link_dict:
            link_dict[gene.output]["incoming"].append((gene.input, gene.weight))
        else:
            link_dict[gene.output] = {"incoming": [(gene.input, gene.weight)], "outgoing": []}

        if gene.input in link_dict:
            link_dict[gene.input]["outgoing"].append((gene.output, gene.weight))
        else:
            link_dict[gene.input] = {"incoming": [], "outgoing": [(gene.output, gene.weight)]}

    outs = []
    ins = []

    for key, value in link_dict.items():
        if len(value["outgoing"]) == 0:
            outs.append(key)
        if len(value["incoming"]) == 0:
            ins.append(key)

    return construct_graph(link_dict, ins, outs)
