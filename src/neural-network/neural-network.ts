import { addLinks, createLayers } from "./functions";
import ImageData from "./data/image-data.json";
import { Batch } from "./types";

const learningRate = 0.1;
const initialInput = (val: number) => (val * 0.99) / 255 + 0.01;

export class Network {
  layers: (Neuron | Input)[][];
  batch: Batch;
  constructor() {
    this.layers = addLinks(createLayers());
    this.batch = ImageData as Batch;
  }
  initializeSample(batchIndex: number) {
    const sample = this.batch[batchIndex];
    (this.layers[0] as Input[]).forEach((input, i) => {
      input.setInputValue(this.batch[batchIndex].pixels[i]);
    });
    (this.layers[3] as Neuron[]).forEach(
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
    (this.layers[3] as Neuron[]).forEach((neuron) => {
      neuron.calculateNewBiasAndWeights(learningRate);
    });
  }
}

export class Link {
  weight: number;
  left?: Neuron | Input;
  right?: Neuron;
  id: string;
  constructor(id: string, weight: number, left: Neuron | Input, right: Neuron) {
    this.weight = weight;
    this.left = left;
    this.right = right;
    this.id = id;
  }
}

export class Input {
  activation?: any;
  links: { right: Link[] };
  id: string;
  constructor(id: string) {
    this.links = { right: [] };
    this.id = id;
  }
  setInputValue(value: number) {
    this.activation = initialInput(value);
  }
}

export class Neuron {
  activation?: number;
  awb?: number;
  bias: number;
  links: { left: Link[]; right?: Link[] };
  expectedOutput?: number;
  cost?: number;
  id: string;
  constructor(id: string, bias: number) {
    this.activation = 0;
    this.bias = bias;
    this.links = { left: [], right: [] };
    this.id = id;
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

      const lR = learningRate;
      const sigmoid = 1 / (1 + Math.exp(-neuron.awb!));

      const dA_dAwb = (1 / sigmoid) * (1 - sigmoid);
      const dAwb_dB = 1;
      const dC_dB = prevCostDerivate * dA_dAwb * dAwb_dB;
      neuron.bias = neuron.bias - lR * dC_dB;
      // (neuron as any).DC_DB = dC_dB;

      neuron.links.left.forEach((link) => {
        const dAwb_dW = link.weight;
        let dC_dW = prevCostDerivate * dA_dAwb * dAwb_dW;
        link.weight = link.weight - lR * dC_dW;
        // (link as any).DC_DW = dC_dW;
        updateAllLeft(link.left as Neuron, dC_dW);
      });
    }
  }
}
