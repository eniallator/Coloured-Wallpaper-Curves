const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const mouse = new Mouse(canvas);
const paramConfig = new ParamConfig(
  "./config.json",
  document.querySelector("#cfg-outer")
);
paramConfig.addCopyToClipboardHandler("#share-btn");

document.getElementById("download-btn").onclick = function (_evt) {
  const url = canvas.toDataURL();
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${
    document.getElementsByTagName("title")?.[0].innerText ?? "download"
  }.png`;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
};

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

ctx.fillStyle = "black";
ctx.strokeStyle = "white";

let pixels;

function draw() {
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const colours = paramConfig.getVal("colours").map((arr) => hexToRGB(arr[0]));
  const shape = [canvas.height, canvas.width, 1];

  let blendSideLength = Math.floor(
    paramConfig.getVal("blend") * Math.min(shape[0], shape[1]) * 0.2
  );
  blendSideLength = blendSideLength + (blendSideLength % 2);
  let shadingFilterSideLength = Math.floor(
    paramConfig.getVal("shadow-size") * Math.min(shape[0], shape[1]) * 0.1
  );
  shadingFilterSideLength =
    shadingFilterSideLength + (shadingFilterSideLength % 2) + 1;
  console.log(shadingFilterSideLength);

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

    const spread = paramConfig.getVal("spread");
    const colourIndices = tf.tidy(() => {
      let colourIndices = tf.zeros(shape);
      for (let i = 0; i < colours.length - 1; i++) {
        // Created here: https://www.desmos.com/calculator/4txy20rwzy
        colourIndices = tf.tidy(() =>
          tf.keep(
            colourIndices.where(
              xCoords
                .sub(0.5)
                .pow(3)
                .mul(4)
                .add(0.5)
                .pow(
                  Math.log(
                    (spread * (i + 1)) / colours.length + (1 - spread) / 2
                  ) / Math.log(1 / 2)
                )
                .less(yCoords),
              colourIndices.add(1)
            )
          )
        );
      }
      return tf.keep(colourIndices);
    });

    const colourChannels = new Array(3).fill().map(() => tf.zeros(shape));

    for (let i = 0; i < colours.length; i++) {
      for (let j = 0; j < colourChannels.length; j++) {
        colourChannels[j] = colourChannels[j].where(
          colourIndices.notEqual(i),
          colours[i][j]
        );
      }
    }

    if (blendSideLength > 1) {
      for (let i = 0; i < colourChannels.length; i++) {
        colourChannels[i] = colourChannels[i]
          .pad(
            [
              [blendSideLength / 2, 0],
              [blendSideLength / 2, 0],
              [0, 0],
            ],
            colours[0][i]
          )
          .pad(
            [
              [0, blendSideLength / 2],
              [0, blendSideLength / 2],
              [0, 0],
            ],
            colours[colours.length - 1][i]
          )
          .conv2d(
            tf.ones([blendSideLength + 1, blendSideLength + 1, 1, 1]),
            1,
            "valid"
          )
          .div(blendSideLength ** 2)
          .minimum(tf.ones([1]));
      }
    }

    if (shadingFilterSideLength > 1) {
      const shading = tf.tidy(() => {
        let shaded = tf.zeros(shape);
        const xCoords = tf
          .range(0, shadingFilterSideLength)
          .sub((shadingFilterSideLength - 1) / 2)
          .tile([shadingFilterSideLength])
          .reshape([shadingFilterSideLength, shadingFilterSideLength]);
        const xyCoords = xCoords.add(
          xCoords.transpose().reshape(xCoords.shape)
        );
        const rotatedXyCoords = xCoords
          .sub(xCoords.transpose().reshape(xCoords.shape))
          .abs();
        const filter = xyCoords
          .where(xyCoords.equal(0), xyCoords.div(xyCoords.abs()))
          .mul(
            tf
              .sub(shadingFilterSideLength, xyCoords.abs())
              .where(
                xyCoords.abs().greater(rotatedXyCoords.abs()),
                tf.sub(shadingFilterSideLength, rotatedXyCoords.abs())
              )
          )
          .reshape([shadingFilterSideLength, shadingFilterSideLength, 1, 1]);
        const filterMaxValue = filter.abs().sum().div(2);
        const paddedIndices = colourIndices
          .pad(
            [
              [(shadingFilterSideLength - 1) / 2, 0],
              [(shadingFilterSideLength - 1) / 2, 0],
              [0, 0],
            ],
            0
          )
          .pad(
            [
              [0, (shadingFilterSideLength - 1) / 2],
              [0, (shadingFilterSideLength - 1) / 2],
              [0, 0],
            ],
            colours.length - 1
          );
        for (let i = 0; i < colours.length; i++) {
          shaded = shaded.add(
            paddedIndices
              .equal(i)
              .cast(paddedIndices.dtype)
              .conv2d(filter, 1, "valid")
              .div(filterMaxValue)
              .where(colourIndices.equal(i), tf.zeros(shape))
          );
        }
        return tf.keep(shaded.add(2).div(2));
      });
      for (let i = 0; i < colourChannels.length; i++) {
        colourChannels[i] = colourChannels[i].mul(shading).minimum(1);
      }
    }

    const borderSize = paramConfig.getVal("border-size");
    const borderMax = paramConfig.getVal("border-max");
    if (borderSize > 0) {
      const xBorders = tf.minimum(
        xCoords.div(borderSize).add(borderMax).minimum(1),
        tf.ones(shape).sub(xCoords).div(borderSize).add(borderMax).minimum(1)
      );
      const yBorders = tf.minimum(
        yCoords.div(borderSize).add(borderMax).minimum(1),
        tf.ones(shape).sub(yCoords).div(borderSize).add(borderMax).minimum(1)
      );
      const borders = xBorders.minimum(yBorders);
      for (let i = 0; i < colourChannels.length; i++) {
        colourChannels[i] = colourChannels[i].mul(borders);
      }
    }

    return tf.keep(
      tf.stack(colourChannels, -1).reshape([shape[0], shape[1], 3])
    );
  });

  tf.browser.toPixels(pixels, canvas);
}

window.onresize = (evt) => {
  const cfgWidth = paramConfig.loaded && paramConfig.getVal("width");
  const cfgHeight = paramConfig.loaded && paramConfig.getVal("height");
  canvas.classList.toggle("full-width", (cfgWidth ?? 0) === 0);
  canvas.classList.toggle("full-height", (cfgHeight ?? 0) === 0);
  canvas.width = cfgWidth || $("#canvas").width();
  canvas.height = cfgHeight || $("#canvas").height();
  draw();
};

paramConfig.addListener(window.onresize, ["redraw"]);

paramConfig.onLoad(window.onresize);
