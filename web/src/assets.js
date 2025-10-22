const manifest = {
  ship: "../assets/ship_01.png",
  enemy: "../assets/enemy.png",
};

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = url;
  });
}

export async function loadAssets() {
  const entries = await Promise.all(
    Object.entries(manifest).map(async ([key, url]) => {
      const image = await loadImage(url);
      return [key, image];
    })
  );
  return Object.fromEntries(entries);
}

export function resizeImage(image, targetHeight) {
  const ratio = targetHeight / image.height;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(image.width * ratio);
  canvas.height = Math.round(image.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  const resized = new Image();
  resized.src = canvas.toDataURL();
  return resized;
}
