import { Input, Neuron, Output } from './neural-network';

interface LayerModel {
  type: string;
  amountOfNodes: number;
}

export type Node = Input | Neuron | Output;

export type NetworkModel = LayerModel[];
