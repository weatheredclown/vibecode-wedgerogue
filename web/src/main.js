import { loadAssets, resizeImage } from "./assets.js";
import { InputManager } from "./input.js";
import { SfxSystem } from "./audio.js";
import { Game } from "./game.js";

async function bootstrap() {
  const canvas = document.getElementById("gameCanvas");
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;

  const input = new InputManager();
  const sfx = new SfxSystem();
  const assets = await loadAssets();
  assets.ship = resizeImage(assets.ship, 40);
  assets.enemy = resizeImage(assets.enemy, 40);

  const game = new Game(ctx, assets, input, sfx);

  function frame() {
    game.update();
    game.draw();
    input.tick();
    requestAnimationFrame(frame);
  }

  frame();
}

bootstrap().catch((err) => {
  console.error("Failed to start Wedgerogue JS", err);
});
