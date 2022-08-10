import { NetworkModel } from './interfaces';

export const networkModel: NetworkModel = [
  {
    type: 'input',
    amountOfNodes: 768,
  },
  { type: 'hiddenLayer', amountOfNodes: 16 },
  { type: 'hiddenLayer', amountOfNodes: 16 },
  { type: 'output', amountOfNodes: 10 },
];

export const learningRate = 0.1;

export const inputs = new Array(768)
  .fill(undefined)
  .map((n) => Math.floor(Math.random() * 10));
export const desiredOutput = 5;
