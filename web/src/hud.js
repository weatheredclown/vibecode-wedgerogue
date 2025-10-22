import { COLORS, WIDTH, HEIGHT, FPS } from "./constants.js";

export class FancyCountdown {
  constructor(fps = FPS, baseFontSize = 100, color = "#ffffff", font = "bold 100px sans-serif") {
    this.fps = fps;
    this.baseFontSize = baseFontSize;
    this.color = color;
    this.font = font;

    this.active = false;
    this.framesLeft = 0;
    this.lastSecond = -1;
  }

  start(totalSeconds = 10) {
    this.active = true;
    this.framesLeft = totalSeconds * this.fps;
    this.lastSecond = Math.ceil(this.framesLeft / this.fps);
  }

  stop() {
    this.active = false;
    this.framesLeft = 0;
    this.lastSecond = -1;
  }

  update() {
    if (!this.active) return;
    this.framesLeft -= 1;
    if (this.framesLeft <= 0) {
      this.stop();
    }
  }

  draw(ctx) {
    if (!this.active) return;
    const secondsLeft = Math.max(1, Math.ceil(this.framesLeft / this.fps));
    const progressWithinSecond = (this.framesLeft % this.fps) / this.fps;
    const scale = 1.5 - progressWithinSecond * 0.5;
    const alpha = Math.min(1, 1 - progressWithinSecond + 0.2);

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = this.color;
    ctx.font = `bold ${Math.round(this.baseFontSize * scale)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${secondsLeft}`, WIDTH / 2, HEIGHT / 2);
    ctx.restore();
  }
}

export class HUDSystem {
  constructor() {
    this.padding = 28;
    this.barWidth = 200;
    this.barHeight = 14;
    this.font = "bold 18px 'Bahnschrift', sans-serif";
    this.smallFont = "bold 14px 'Bahnschrift', sans-serif";
  }

  draw(ctx, player) {
    ctx.save();
    ctx.font = this.font;
    ctx.fillStyle = COLORS.text;

    this.drawBar(ctx, player.hp, player.maxHp, this.padding, this.padding, COLORS.hp, "HP");
    let y = this.padding + this.barHeight + this.padding;
    if (player.shieldMax > 0) {
      this.drawBar(ctx, player.shieldHp, player.shieldMax, this.padding, y, COLORS.shield, "Shield");
      y += this.barHeight + this.padding;
    }

    const scoreText = `Score: ${Math.floor(player.score)}`;
    ctx.textAlign = "right";
    ctx.fillStyle = COLORS.score;
    ctx.fillText(scoreText, WIDTH - this.padding, this.padding + 12);

    ctx.restore();
  }

  drawBar(ctx, current, max, x, y, color, label) {
    ctx.save();
    ctx.fillStyle = COLORS.barBg;
    ctx.fillRect(x, y, this.barWidth, this.barHeight);
    ctx.strokeStyle = COLORS.barOutline;
    ctx.strokeRect(x - 1, y - 1, this.barWidth + 2, this.barHeight + 2);

    const fraction = max > 0 ? Math.max(0, Math.min(1, current / max)) : 0;
    const width = this.barWidth * fraction;
    ctx.fillStyle = color;
    ctx.fillRect(x, y, width, this.barHeight);

    ctx.fillStyle = COLORS.text;
    ctx.font = this.smallFont;
    ctx.textAlign = "left";
    ctx.fillText(label, x, y - 6);
    ctx.textAlign = "center";
    ctx.fillText(`${Math.floor(current)}/${Math.floor(max)}`, x + this.barWidth / 2, y + this.barHeight / 2 + 5);
    ctx.restore();
  }
}
