import { addLinks, createLayers } from "./functions";
import ImageData from "./data/image-data.json";
import { Batch } from "./types";

const learningRate = 0.5;
const initialInput = (val: number) => (val * 0.99) / 255 + 0.01;
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export class Network {
  layers: (Neuron | Input)[][];
  batch: Batch;
  constructor() {
    this.layers = addLinks(createLayers());
    this.batch = ImageData as Batch;
  }
  get outputLayer() {
    return this.layers[this.layers.length - 1] as Neuron[];
  }
  initializeSample(batchIndex: number) {
    const sample = this.batch[batchIndex];
    (this.layers[0] as Input[]).forEach((input, i) => {
      input.setInputValue(sample.pixels[i]);
    });
    this.outputLayer.forEach(
      (neuron, i) => (neuron.expectedOutput = i === sample.label ? 0.99 : 0.01)
    );
  }
  propagateForward() {
    this.layers.forEach((layer, i) => {
      if (i === 0) return;
      (layer as Neuron[]).forEach((neuron) => {
        neuron.calculateActivation();
        if (i === this.layers.length - 1) neuron.calculateCost();
      });
    });
  }
  propagateBack() {
    // 1. Compute every neuron's error term (delta = dC/dz), starting at the
    //    output layer and moving left. This must happen before any weight is
    //    changed, since each hidden delta depends on the weights to its right.
    for (let i = this.layers.length - 1; i > 0; i--) {
      (this.layers[i] as Neuron[]).forEach((neuron) => neuron.calculateDelta());
    }
    // 2. Apply gradient descent to every bias and incoming weight.
    for (let i = 1; i < this.layers.length; i++) {
      (this.layers[i] as Neuron[]).forEach((neuron) =>
        neuron.updateBiasAndWeights(learningRate)
      );
    }
  }
  /** Index of the output neuron with the highest activation. */
  guessedDigit() {
    let best = 0;
    this.outputLayer.forEach((neuron, i) => {
      if (neuron.activation > this.outputLayer[best].activation) best = i;
    });
    return best;
  }
  /** Trains once on every sample in random order. Returns the share guessed
   *  correctly (measured on each sample just before training on it). */
  runEpoch() {
    const order = this.batch.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    let correct = 0;
    order.forEach((index) => {
      this.initializeSample(index);
      this.propagateForward();
      if (this.guessedDigit() === this.batch[index].label) correct++;
      this.propagateBack();
    });
    return correct / order.length;
  }
}

export class Link {
  weight: number;
  left: Neuron | Input;
  right: Neuron;
  id: string;
  constructor(id: string, weight: number, left: Neuron | Input, right: Neuron) {
    this.weight = weight;
    this.left = left;
    this.right = right;
    this.id = id;
  }
}

export class Input {
  activation: number;
  links: { right: Link[] };
  id: string;
  constructor(id: string) {
    this.activation = 0;
    this.links = { right: [] };
    this.id = id;
  }
  setInputValue(value: number) {
    this.activation = initialInput(value);
  }
}

export class Neuron {
  activation: number;
  awb?: number;
  bias: number;
  /** dC/d(awb): how much the cost changes with this neuron's weighted sum. */
  delta: number;
  links: { left: Link[]; right?: Link[] };
  expectedOutput?: number;
  cost?: number;
  id: string;
  constructor(id: string, bias: number) {
    this.activation = 0;
    this.bias = bias;
    this.delta = 0;
    this.links = { left: [], right: [] };
    this.id = id;
  }

  calculateActivation() {
    let aw = 0;
    this.links.left.forEach((link) => {
      aw += link.weight * link.left.activation;
    });
    this.awb = aw + this.bias;
    this.activation = sigmoid(this.awb);
  }
  calculateCost() {
    this.cost = Math.pow(this.activation - this.expectedOutput!, 2);
  }
  calculateDelta() {
    // dA/d(awb) for a sigmoid is σ(awb)·(1 − σ(awb)), i.e. a·(1 − a).
    const dA_dAwb = this.activation * (1 - this.activation);

    let dC_dA: number;
    if (this.expectedOutput !== undefined) {
      // Output neuron: C = (a − y)²
      dC_dA = 2 * (this.activation - this.expectedOutput);
    } else {
      // Hidden neuron: its activation feeds every neuron to its right, so its
      // effect on the cost is the sum over those links of weight · delta.
      dC_dA = 0;
      this.links.right!.forEach((link) => {
        dC_dA += link.weight * link.right.delta;
      });
    }
    this.delta = dC_dA * dA_dAwb;
  }
  updateBiasAndWeights(learningRate: number) {
    // d(awb)/dB = 1
    this.bias -= learningRate * this.delta;
    this.links.left.forEach((link) => {
      // d(awb)/dW = activation of the neuron on the left side of the link
      link.weight -= learningRate * this.delta * link.left.activation;
    });
  }
}
