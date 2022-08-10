import * as fs from 'fs';
import { createCanvas } from 'canvas';

function readMNIST(start, end) {
  const dataFileBuffer = fs.readFileSync(
    'src/neural-network/data/train-images.idx3-ubyte'
  );
  const labelFileBuffer = fs.readFileSync(
    'src/neural-network/data/train-labels.idx1-ubyte'
  );
  let pixelValues = [];

  for (let image = start; image < end; image++) {
    let pixels = [];
    for (let y = 0; y <= 27; y++) {
      for (let x = 0; x <= 27; x++) {
        pixels.push(dataFileBuffer[image * 28 * 28 + (x + y * 28) + 16]);
      }
    }

    let imageData = {};
    imageData['index'] = image;
    imageData['label'] = labelFileBuffer[image + 8];
    imageData['pixels'] = pixels;
    pixelValues.push(imageData);
  }
  fs.writeFileSync(
    `src/neural-network/data/image-data.json`,
    JSON.stringify(pixelValues)
  );
  return pixelValues;
}

function saveMNIST(start, end) {
  const canvas = createCanvas(28, 28);
  const ctx = canvas.getContext('2d');

  const pixelValues = readMNIST(start, end);

  pixelValues.forEach(function (image) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y <= 27; y++) {
      for (let x = 0; x <= 27; x++) {
        const pixel = image.pixels[x + y * 28];
        const colour = 255 - pixel;
        ctx.fillStyle = `rgb(${colour}, ${colour}, ${colour})`;
        ctx.fillRect(x, y, 1, 1);
      }
    }
    const buffer = canvas.toBuffer('image/png');
    fs.writeFileSync(
      `src/neural-network/data/images/image${image.index}-${image.label}.png`,
      buffer
    );
  });
}

saveMNIST(0, 10);
