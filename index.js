const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const mouse = new Mouse(canvas);
const paramConfig = new ParamConfig(
  "./config.json",
  document.querySelector("#cfg-outer")
);
paramConfig.addCopyToClipboardHandler("#share-btn");

function hexToRGB(hex) {
  const match = hex
    .toUpperCase()
    .match(/^#?([\dA-F]{2})([\dA-F]{2})([\dA-F]{2})$/);
  return [
    parseInt(match[1], 16) / 255,
    parseInt(match[2], 16) / 255,
    parseInt(match[3], 16) / 255,
  ];
}

window.onresize = (evt) => {
  canvas.width = $("#canvas").width();
  canvas.height = $("#canvas").height();
};
window.onresize();

ctx.fillStyle = "black";
ctx.strokeStyle = "white";

let pixels;

function draw() {
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const colours = paramConfig.getVal("colours").map((arr) => hexToRGB(arr[0]));
  const shape = [canvas.height, canvas.width, 1];

  let blendSideLength = Math.floor(
    paramConfig.getVal("blend") * Math.min(canvas.width, canvas.height) * 0.2
  );
  blendSideLength = blendSideLength + (blendSideLength % 2);

  if (pixels != null) {
    pixels.dispose();
  }

  pixels = tf.tidy(() => {
    const xCoords = tf
      .range(0, shape[1])
      .tile([shape[0]])
      .reshape(shape)
      .div(shape[1]);
    const yCoords = tf
      .range(shape[0], 0)
      .reshape([shape[0], 1])
      .tile([1, shape[1]])
      .reshape(shape)
      .div(shape[0]);

    let colourIndices = tf.zeros(shape);
    for (let i = 0; i < colours.length; i++) {
      // Created here: https://www.desmos.com/calculator/bsei7grt6h
      colourIndices = colourIndices.where(
        xCoords
          .sub(0.5)
          .pow(3)
          .mul(4)
          .add(0.5)
          .pow((paramConfig.getVal("spread") * i) / colours.length)
          .less(yCoords),
        colourIndices.add(1)
      );
    }

    let r = tf.zeros(shape);
    let g = tf.zeros(shape);
    let b = tf.zeros(shape);

    for (let i = 0; i < colours.length; i++) {
      r = r.where(colourIndices.notEqual(i + 1), colours[i][0]);
      g = g.where(colourIndices.notEqual(i + 1), colours[i][1]);
      b = b.where(colourIndices.notEqual(i + 1), colours[i][2]);
    }

    r = r
      .pad(
        [
          [blendSideLength / 2, 0],
          [blendSideLength / 2, 0],
          [0, 0],
        ],
        colours[0][0]
      )
      .pad(
        [
          [0, blendSideLength / 2],
          [0, blendSideLength / 2],
          [0, 0],
        ],
        colours[colours.length - 1][0]
      )
      .conv2d(
        tf.ones([blendSideLength + 1, blendSideLength + 1, 1, 1]),
        1,
        "valid"
      )
      .div(blendSideLength ** 2)
      .minimum(tf.ones([1]));
    g = g
      .pad(
        [
          [blendSideLength / 2, 0],
          [blendSideLength / 2, 0],
          [0, 0],
        ],
        colours[0][1]
      )
      .pad(
        [
          [0, blendSideLength / 2],
          [0, blendSideLength / 2],
          [0, 0],
        ],
        colours[colours.length - 1][1]
      )
      .conv2d(
        tf.ones([blendSideLength + 1, blendSideLength + 1, 1, 1]),
        1,
        "valid"
      )
      .div(blendSideLength ** 2)
      .minimum(tf.ones([1]));
    b = b
      .pad(
        [
          [blendSideLength / 2, 0],
          [blendSideLength / 2, 0],
          [0, 0],
        ],
        colours[0][2]
      )
      .pad(
        [
          [0, blendSideLength / 2],
          [0, blendSideLength / 2],
          [0, 0],
        ],
        colours[colours.length - 1][2]
      )
      .conv2d(
        tf.ones([blendSideLength + 1, blendSideLength + 1, 1, 1]),
        1,
        "valid"
      )
      .div(blendSideLength ** 2)
      .minimum(tf.ones([1]));

    return tf.keep(tf.stack([r, g, b], 2).reshape([shape[0], shape[1], 3]));
  });

  tf.browser.toPixels(pixels, canvas);
}

paramConfig.addListener(draw, ["redraw"]);

paramConfig.onLoad(draw);
