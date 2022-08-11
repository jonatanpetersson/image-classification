import { Input, Neuron, Output } from './neural-network';

interface LayerModel {
  type: 'input' | 'hiddenLayer' | 'output';
  amountOfNodes: number;
}

export interface Image {
  index: number;
  label: number;
  pixels: number[];
}
export type Node = Input | Neuron | Output;
export type NetworkModel = LayerModel[];
export type Images = Image[];
