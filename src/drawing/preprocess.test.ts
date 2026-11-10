import { MNIST_SIZE, toMnistPixels } from './preprocess';
import ImageData from '../neural-network/data/image-data.json';

const upscale = (pixels: number[], factor: number) => {
  const size = MNIST_SIZE * factor;
  const out = new Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      out[y * size + x] =
        pixels[Math.floor(y / factor) * MNIST_SIZE + Math.floor(x / factor)];
    }
  }
  return out;
};

test('blank drawing gives all zeros', () => {
  const out = toMnistPixels(new Array(280 * 280).fill(0), 280, 280);
  expect(out).toHaveLength(784);
  expect(out.every(v => v === 0)).toBe(true);
});

test('scales a small off-center shape to a 20 pixel box around the center', () => {
  const size = 280;
  const gray = new Array(size * size).fill(0);
  // A 10×40 bar in the top-left corner.
  for (let y = 5; y < 45; y++) for (let x = 5; x < 15; x++) gray[y * size + x] = 255;
  const out = toMnistPixels(gray, size, size);
  const rows = new Set<number>(), cols = new Set<number>();
  out.forEach((v, i) => {
    if (v > 0) {
      rows.add(Math.floor(i / MNIST_SIZE));
      cols.add(i % MNIST_SIZE);
    }
  });
  expect(rows.size).toBe(20);
  expect(cols.size).toBe(5);
  // Shifted so the center of mass (row 9.5, column 2 of the 5×20 bar) lands
  // on pixel 14: 14 − 9.5 = 4.5 rounds to 5, and 14 − 2 = 12.
  expect(Math.min(...Array.from(rows))).toBe(5);
  expect(Math.min(...Array.from(cols))).toBe(12);
});

test('MNIST digits drawn 10× larger come back almost unchanged', () => {
  const samples = (ImageData as { pixels: number[] }[]).slice(0, 100);
  let identical = 0;
  samples.forEach(({ pixels }) => {
    const out = toMnistPixels(upscale(pixels, 10), 280, 280);
    if (out.every((v, i) => v === pixels[i])) identical++;
  });
  expect(identical).toBeGreaterThanOrEqual(85);
});
