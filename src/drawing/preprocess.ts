export const MNIST_SIZE = 28;
// MNIST digits are scaled to fit a 20×20 box, then placed in the 28×28 frame
// so that their center of mass sits in the middle.
const DIGIT_BOX = 20;

/**
 * Converts a grayscale drawing (0 = background, 255 = ink, row by row) of any
 * size into 28×28 pixels prepared the same way as the MNIST images.
 */
export function toMnistPixels(
  gray: ArrayLike<number>,
  width: number,
  height: number
): number[] {
  const out: number[] = Array(MNIST_SIZE * MNIST_SIZE).fill(0);

  // 1. Bounding box of everything drawn.
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (gray[y * width + x] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return out;

  // 2. Scale the box so its longest side is 20 pixels, keeping the aspect
  //    ratio. Each new pixel is the average of the source pixels it covers.
  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const scale = DIGIT_BOX / Math.max(boxW, boxH);
  const w = Math.max(1, Math.round(boxW * scale));
  const h = Math.max(1, Math.round(boxH * scale));
  const span = (t: number, min: number, max: number) => {
    const from = Math.min(max, min + Math.floor(t / scale));
    const to = Math.min(max + 1, Math.max(from + 1, min + Math.floor((t + 1) / scale)));
    return [from, to];
  };
  const small: number[] = Array(w * h).fill(0);
  for (let ty = 0; ty < h; ty++) {
    const [y0, y1] = span(ty, minY, maxY);
    for (let tx = 0; tx < w; tx++) {
      const [x0, x1] = span(tx, minX, maxX);
      let sum = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) sum += gray[y * width + x];
      }
      small[ty * w + tx] = sum / ((y1 - y0) * (x1 - x0));
    }
  }

  // 3. Shift so the center of mass lands on pixel (14, 14), where MNIST
  //    puts it.
  let mass = 0, comX = 0, comY = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = small[y * w + x];
      mass += v;
      comX += v * x;
      comY += v * y;
    }
  }
  const offsetX = Math.round(MNIST_SIZE / 2 - comX / mass);
  const offsetY = Math.round(MNIST_SIZE / 2 - comY / mass);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const X = x + offsetX, Y = y + offsetY;
      if (X >= 0 && X < MNIST_SIZE && Y >= 0 && Y < MNIST_SIZE) {
        out[Y * MNIST_SIZE + X] = Math.round(small[y * w + x]);
      }
    }
  }
  return out;
}
