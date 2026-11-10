import { v4 as uuid } from "uuid";
import { Input, Link, Neuron } from "./neural-network";

const inputSize = 28 * 28;
const layerSizes = [16, 16, 10];

/** Uniform random value in [-1/√fanIn, 1/√fanIn], so neurons start out
 *  different from each other and their weighted sums stay in sigmoid's
 *  responsive range. */
const randomWeight = (fanIn: number) =>
  (Math.random() * 2 - 1) / Math.sqrt(fanIn);

export function createLayers() {
  const inputs = Array(inputSize)
    .fill(null)
    .map(() => new Input(uuid()));
  const layers: (Neuron | Input)[][] = layerSizes.map((num) =>
    Array(num)
      .fill(null)
      .map(() => new Neuron(uuid(), 0))
  );
  layers.unshift(inputs);
  return layers;
}

export function addLinks(layers: (Neuron | Input)[][]) {
  layers.forEach((layerLeft, i) => {
    if (i === layers.length - 1) {
      layers[i].forEach((output) => {
        delete (output as Neuron).links.right;
      });
      return;
    }
    const layerRight = layers[i + 1];
    layerLeft.forEach((neuronLeft) => {
      layerRight.forEach((neuronRight) => {
        const link = new Link(
          uuid(),
          randomWeight(layerLeft.length),
          neuronLeft,
          neuronRight as Neuron
        );
        neuronLeft.links.right!.push(link);
        (neuronRight as Neuron).links.left.push(link);
      });
    });
  });
  return layers;
}
