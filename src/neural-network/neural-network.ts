import { inputs } from './constants';
import { NetworkModel, Node } from './interfaces';

export class Network {
  layers: Layer[];
  connections: Connection[];
  learningRate: number;
  constructor(networkModel: NetworkModel, learningRate: number) {
    this.learningRate = learningRate;
    this.layers = this.createLayers(networkModel);
    this.connections = this.createConnections(this.layers);
  }

  iterate(input: number[], desiredOutput: number) {}

  createLayers(networkModel: NetworkModel): Layer[] {
    return networkModel.map(
      (layer) => new Layer(layer.type, layer.amountOfNodes, this.learningRate)
    );
  }

  createConnections(layers: Layer[]): Connection[] {
    let connections: Connection[] = [];
    layers.forEach((layerLeft, i) => {
      if (i === layers.length - 1) {
        return;
      }
      const layerRight = layers[i + 1];
      layerLeft.nodes!.forEach((nodeLeft) => {
        layerRight.nodes!.forEach((nodeRight) => {
          const connection = new Connection(
            Math.random(),
            nodeLeft,
            nodeRight,
            this.learningRate
          );
          connections.push(connection);

          if (!(nodeLeft instanceof Output)) {
            nodeLeft.setConnectionRight(connection);
          }
          if (!(nodeRight instanceof Input)) {
            nodeRight.setConnectionLeft(connection);
          }
        });
      });
    });
    return connections;
  }
}

export class Layer {
  type: string;
  learningRate: number;
  nodes?: Neuron[] | Input[] | Output[];
  constructor(type: string, amountOfNodes: number, learningRate: number) {
    this.type = type;
    this.learningRate = learningRate;
    this.nodes = this.createNodes(amountOfNodes);
  }
  createNodes(amountOfNodes: number): Neuron[] | Input[] | Output[] {
    let node: any;
    switch (this.type) {
      case 'input':
        node = (input: number) => new Input(input);
        break;
      case 'output':
        node = () => new Output();
        break;
      case 'hiddenLayer':
        node = () => new Neuron(Math.random(), this.learningRate);
        break;
    }
    return new Array(amountOfNodes)
      .fill(undefined)
      .map((n, i) => (this.type === 'input' ? node(inputs[i]) : node()));
  }
}

export class Input {
  value: number;
  connectionRight?: Connection;
  constructor(value: number) {
    this.value = value;
  }
  setConnectionRight(connectionRight: Connection) {
    this.connectionRight = connectionRight;
  }
}

export class Output {
  connectionLeft?: Connection;
  output?: number;
  constructor() {}
  setConnectionLeft(connectionLeft: Connection) {
    this.connectionLeft = connectionLeft;
  }
}

export class Neuron {
  learningRate: number;
  bias?: number;
  desiredOutput?: number;
  connectionLeft?: Connection;
  connectionRight?: Connection;
  input?: number;
  activation?: number;
  gradientDescent?: number;
  constructor(bias: number, learningRate: number) {
    this.bias = bias;
    this.learningRate = learningRate;
  }

  // a = f(max(0, x*w+b))
  calculateActivation(input: number, weight: number, bias: number): number {
    const xwb = input * weight + bias;
    return xwb > 0 ? xwb : 0;
  }

  // c = (a - y^2)
  calculateCost() {
    return this.activation! - Math.pow(this.desiredOutput!, 2);
  }

  calculateGradientDescent() {
    // dc = 2(a - y)
    const deltaCost = 2 * (this.activation! - this.desiredOutput!);
    // da = input
    const deltaActivation = this.input;
    this.gradientDescent = deltaCost * deltaActivation!;
  }

  setConnectionLeft(connectionLeft: Connection) {
    this.connectionLeft = connectionLeft;
  }
  setConnectionRight(connectionRight: Connection) {
    this.connectionRight = connectionRight;
  }
  setdesiredOutput(desiredOutput: number) {
    this.desiredOutput = desiredOutput;
  }
}

export class Connection {
  weight: number;
  learningRate: number;
  nodeLeft: Node;
  nodeRight: Node;
  constructor(
    weight: number,
    nodeLeft: Node,
    nodeRight: Node,
    learningRate: number
  ) {
    this.weight = weight;
    this.nodeLeft = nodeLeft;
    this.nodeRight = nodeRight;
    this.learningRate = learningRate;
  }

  // w1 = w0 - r*dc
  calculateNewWeight(dc: number) {
    this.weight = this.weight - this.learningRate * dc;
  }
}

export function createNetwork(
  networkModel: NetworkModel,
  learningRate: number
): Network {
  return new Network(networkModel, learningRate);
}
