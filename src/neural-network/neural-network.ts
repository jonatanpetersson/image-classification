import { imageData } from './constants';
import { Images, NetworkModel, Node } from './types';

export class Network {
  layers: Layer[];
  connections: Connection[];
  learningRate: number;
  imageData: Images;
  iteration: number = -1;
  constructor(
    networkModel: NetworkModel,
    learningRate: number,
    imageData: Images
  ) {
    this.learningRate = learningRate;
    this.imageData = imageData;
    this.layers = this.createLayers(networkModel);
    this.connections = this.createConnections(this.layers);
    console.time('timer');
    this.addConnectionsToNodes();
    console.timeEnd('timer');
  }

  iterate() {
    this.iteration += this.iteration === 9 ? -9 : 1;
    this.loadImageData(
      imageData[this.iteration].pixels,
      imageData[this.iteration].label
    );
    this.layers[1].nodes.forEach((neuron) => {
      (neuron as Neuron).calculateActivation();
    });
    this.layers[2].nodes.forEach((neuron) => {
      (neuron as Neuron).calculateActivation();
    });
    this.layers[3].nodes.forEach((neuron) => {
      (neuron as Output).calculateActivationAndCost();
    });
    this.layers[2].nodes.forEach((neuron) => {
      (neuron as Neuron).calculateActivation();
    });
    console.log(this);
  }
  loadImageData(inputs: number[], desiredOutput: number) {
    this.layers[0].nodes.forEach((node, i) => {
      (node as Input).input = inputs[i];
    });
    this.layers[3].nodes.forEach((node, i) => {
      (node as Output).desiredOutput = i === desiredOutput ? 1 : 0;
    });
  }

  addConnectionsToNodes() {
    this.layers.forEach((layer, i) => {
      if (i === this.layers.length - 1) {
        return;
      }
      const layerLeft = layer;
      const layerRight = this.layers[i + 1];

      layerLeft.nodes.forEach((nodeLeft) => {
        layerRight.nodes.forEach((nodeRight) => {
          const connection = this.connections.find(
            (c) => c.nodeLeft === nodeLeft && c.nodeRight === nodeRight
          );
          if (!(nodeLeft instanceof Output)) {
            nodeLeft.connectionsRight.push(connection as Connection);
          }
          if (!(nodeRight instanceof Input)) {
            nodeRight.connectionsLeft.push(connection as Connection);
          }
        });
      });
    });
  }

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
        });
      });
    });
    return connections;
  }
}

export class Layer {
  type: string;
  learningRate: number;
  nodes: Neuron[] | Input[] | Output[];
  constructor(type: string, amountOfNodes: number, learningRate: number) {
    this.type = type;
    this.learningRate = learningRate;
    this.nodes = this.createNodes(amountOfNodes);
  }
  createNodes(amountOfNodes: number): Neuron[] | Input[] | Output[] {
    let node: any;
    switch (this.type) {
      case 'input':
        node = (input: number): Input => new Input();
        break;
      case 'output':
        node = (): Output => new Output(Math.random());
        break;
      case 'hiddenLayer':
        node = (): Neuron => new Neuron(Math.random(), this.learningRate);
        break;
    }
    return new Array(amountOfNodes)
      .fill(undefined)
      .map((n, i) => (this.type === 'input' ? node() : node()));
  }
}

export class Input {
  input?: number;
  connectionsRight: Connection[] = [];
  constructor() {}
  setInput(input: number) {
    this.input = input;
  }
}

export class Output {
  bias: number;
  connectionsLeft: Connection[] = [];
  cost?: number;
  desiredOutput?: number;
  output?: number;
  deltaC?: number;
  constructor(bias: number) {
    this.bias = bias;
  }

  // c = (a - y^2)
  calculateCost() {
    this.cost = Math.pow(this.output! - this.desiredOutput!, 2);
  }

  calculateDeltaC() {
    // dc = 2(a - y)
    const deltaCost = 2 * (this.output! - this.desiredOutput!);
    // da = input
    const deltaActivation = this.desiredOutput;
    this.deltaC = deltaCost * deltaActivation!;
  }

  getInputsAndWeights(): { weights: number[]; inputs: number[] } {
    let weights: number[] = [];
    let inputs: number[] = [];
    this.connectionsLeft.forEach((cLeft) => {
      weights.push(cLeft.weight);
      if (cLeft.nodeLeft instanceof Neuron) {
        inputs.push(cLeft.nodeLeft.activation!);
      }
    });
    return { weights, inputs };
  }

  calculateActivationAndCost() {
    const { weights, inputs } = this.getInputsAndWeights();
    let wx = 0;
    for (let i = 0; i < weights.length; i++) {
      wx += weights[i] * inputs[i];
    }
    // const xwb = wx + this.bias;
    const xwb = wx;
    this.output = 1 / (1 + Math.exp(-xwb)); // Sigmoid
    // this.output = xwb > 0 ? xwb : 0; // RELU: a = f(max(0, x*w+b))
    this.calculateCost();
    this.calculateDeltaC();
  }
}

export class Neuron {
  learningRate: number;
  bias: number;
  desiredOutput?: number;
  connectionsLeft: Connection[] = [];
  connectionsRight: Connection[] = [];
  activation?: number; // new input value!!!
  deltaC?: number;
  constructor(bias: number, learningRate: number) {
    this.bias = bias;
    this.learningRate = learningRate;
  }

  getInputsAndWeights(): { weights: number[]; inputs: number[] } {
    let weights: number[] = [];
    let inputs: number[] = [];
    this.connectionsLeft.forEach((cLeft) => {
      weights.push(cLeft.weight);
      if (cLeft.nodeLeft instanceof Input) {
        inputs.push(cLeft.nodeLeft.input!);
      }
      if (cLeft.nodeLeft instanceof Neuron) {
        inputs.push(cLeft.nodeLeft.activation!);
      }
    });
    return { weights, inputs };
  }

  calculateActivation() {
    const { weights, inputs } = this.getInputsAndWeights();
    let wx = 0;
    for (let i = 0; i < weights.length; i++) {
      wx += weights[i] * inputs[i];
    }
    const xwb = wx + this.bias;
    // this.activation = 1 / (1 + Math.exp(-xwb)); // Sigmoid
    this.activation = xwb > 0 ? xwb : 0; // RELU: a = f(max(0, x*w+b))
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
  calculateNewWeight(deltaC: number) {
    this.weight = this.weight - this.learningRate * deltaC;
  }
}
