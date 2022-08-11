import { Images, NetworkModel } from './types.js';
import imageDataJSON from './data/image-data.json';

export const imageData: Images = JSON.parse(JSON.stringify(imageDataJSON));

export const networkData: NetworkModel = [
  {
    type: 'input',
    amountOfNodes: imageData[0].pixels.length,
  },
  { type: 'hiddenLayer', amountOfNodes: 16 },
  { type: 'hiddenLayer', amountOfNodes: 16 },
  { type: 'output', amountOfNodes: 10 },
];

export const learningRate = 0.1;
