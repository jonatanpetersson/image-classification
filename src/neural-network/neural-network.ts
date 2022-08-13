import { addLinks, createLayers } from "./functions";
import ImageData from "./data/image-data.json";
import { Batch } from "./types";

export class Network {
  layers: (Neuron | Input)[][];
  batch: Batch;
  constructor() {
    this.layers = addLinks(createLayers());
    this.batch = ImageData;
  }
  initializeSample(batchIndex: number) {
    const sample = this.batch[batchIndex];
    (this.layers[0] as Input[]).forEach((input, i) => {
      input.setInputValue(this.batch[batchIndex].pixels[i]);
    });
    (this.layers[3] as Neuron[]).forEach(
      (neuron, i) => (neuron.expectedOutput = i === sample.label ? 1 : 0)
    );
  }
  propagateForward() {
    this.layers.forEach((layer, i) => {
      if (i === 0) return;
      (layer as Neuron[]).forEach((neuron) => {
        neuron.calculateActivation();
        neuron.calculateCost();
      });
    });
  }
  propagateBack() {
    (this.layers[3] as Neuron[]).forEach((neuron) => {
      neuron.calculateNewBiasAndWeights(0.1);
    });
  }
}

export class Link {
  weight: number;
  left?: Neuron | Input;
  right?: Neuron;
  constructor(weight: number, left: Neuron | Input, right: Neuron) {
    this.weight = weight;
    this.left = left;
    this.right = right;
  }
}

export class Input {
  activation?: any;
  links: { right: Link[] };
  constructor() {
    this.links = { right: [] };
  }
  setInputValue(value: number) {
    this.activation = value;
  }
}

export class Neuron {
  activation?: number;
  awb?: number;
  bias: number;
  links: { left: Link[]; right?: Link[] };
  expectedOutput?: number;
  cost?: number;
  constructor(bias: number) {
    this.activation = 0;
    this.bias = bias;
    this.links = { left: [], right: [] };
  }

  calculateActivation() {
    let aw = 0;
    this.links.left.forEach((link) => {
      aw += link.weight * link.left?.activation;
    });
    this.awb = aw + this.bias;
    this.activation = 1 / (1 + Math.exp(-this.awb));
  }
  calculateCost() {
    this.cost = Math.pow(this.activation! - this.expectedOutput!, 2);
  }
  calculateNewBiasAndWeights(learningRate: number) {
    const dC_dA = 2 * (this.activation! - this.expectedOutput!);

    updateAllLeft(this, dC_dA);

    function updateAllLeft(neuron: Neuron, prevCostDerivate: number) {
      if (!neuron.links.left) return;
      const sigmoid = 1 / (1 + Math.exp(-neuron.awb!));
      const lR = learningRate;

      const dA_dAwb = (1 / sigmoid) * (1 - sigmoid);
      const dAwb_dB = 1;
      const dC_dB = prevCostDerivate * dA_dAwb * dAwb_dB;
      neuron.bias = neuron.bias - lR * dC_dB;

      neuron.links.left.forEach((link) => {
        const dAwb_dW = link.weight;
        const dC_dW = prevCostDerivate * dA_dAwb * dAwb_dW;
        link.weight = link.weight - lR * dC_dW;
        updateAllLeft(link.left as Neuron, dC_dW);
      });
    }
  }
}
