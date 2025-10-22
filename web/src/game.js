import { WIDTH, HEIGHT, NUM_ROOMS, COLLECTION_TIME } from "./constants.js";
import { Vector2 } from "./utils.js";
import { Enemy, FancyLootToken, ParticleSystem, PlayerShip } from "./entities.js";
import { FancyCountdown, HUDSystem } from "./hud.js";
import {
  postDialogs,
  pickUnusedPreDialog,
  createSequenceDialog,
  createChoiceDialog,
  applyDialogEffects,
} from "./dialog.js";

class Room {
  constructor(index, assets, playSfx, { isBoss = false } = {}) {
    this.index = index;
    this.isBoss = isBoss;
    this.assets = assets;
    this.playSfx = playSfx;

    this.enemies = [];
    const enemyCount = isBoss ? 1 : Math.max(1, Math.floor(Math.random() * 3) + 1);
    for (let i = 0; i < enemyCount; i++) {
      const pos = new Vector2(
        Math.random() * (WIDTH - 200) + 100,
        Math.random() * (HEIGHT - 200) + 100
      );
      this.enemies.push(new Enemy(pos, assets.enemy, { playSfx, isBoss: isBoss && i === 0 }));
    }

    this.loot = [];
    this.particles = new ParticleSystem();
    this.countdown = new FancyCountdown();
    this.collectionStarted = false;
  }

  startCollection() {
    if (!this.collectionStarted) {
      this.countdown.start(COLLECTION_TIME);
      this.collectionStarted = true;
    }
  }

  update(player, globalFrame) {
    for (const enemy of this.enemies) {
      enemy.update(globalFrame, this.enemies);
    }

    for (let i = this.enemies.length - 1; i >= 0; i--) {
      const enemy = this.enemies[i];
      for (const bullet of player.bullets) {
        if (enemy.hitBy(bullet)) {
          enemy.health -= 1;
          if (this.playSfx) this.playSfx(13);
          if (bullet.exploding) {
            for (const other of this.enemies) {
              if (other === enemy) continue;
              const diff = other.pos.clone().sub(bullet.pos);
              if (diff.length() < 40) other.health -= 1;
            }
          }
          bullet.life = 0;
          if (enemy.health <= 0) {
            if (!enemy.droppedLoot) {
              enemy.droppedLoot = true;
              this.loot.push(new FancyLootToken(enemy.pos.clone(), enemy.color, enemy.points, this.particles));
              player.score += enemy.points;
            }
            this.enemies.splice(i, 1);
            break;
          }
        }
      }
    }

    for (const enemy of this.enemies) {
      for (let i = enemy.bullets.length - 1; i >= 0; i--) {
        const bullet = enemy.bullets[i];
        const radius = bullet.radius + 20;
        const diffX = bullet.pos.x - player.pos.x;
        const diffY = bullet.pos.y - player.pos.y;
        if (diffX * diffX + diffY * diffY < radius * radius) {
          if (player.shieldHp > 0) {
            player.shieldHp = Math.max(0, player.shieldHp - 1);
          } else {
            player.hp = Math.max(0, player.hp - 1);
          }
          if (this.playSfx) this.playSfx(9);
          this.particles.spawnHitParticles(bullet.pos.clone(), [255, 200, 50], 8);
          enemy.bullets.splice(i, 1);
        }
      }
    }

    for (let i = this.loot.length - 1; i >= 0; i--) {
      const token = this.loot[i];
      token.update();
      if (token.checkCollision(player)) {
        token.onCollected(this.particles);
        if (this.playSfx) this.playSfx(27);
        player.score += token.value * 5;
        this.loot.splice(i, 1);
      }
    }

    this.particles.update();
    this.countdown.update();
  }

  draw(ctx) {
    for (const enemy of this.enemies) {
      enemy.draw(ctx);
    }
    for (const token of this.loot) {
      token.draw(ctx);
    }
    this.particles.draw(ctx);
    this.countdown.draw(ctx);
  }

  isEnemyCleared() {
    return this.enemies.length === 0;
  }

  isCollectionDone() {
    return this.collectionStarted && !this.countdown.active;
  }
}

export class Game {
  constructor(ctx, assets, input, sfx) {
    // TODO: Port the in-game store, options menu, and star map travel sequences
    // once the core combat loop is fully verified in JavaScript.
    this.ctx = ctx;
    this.assets = assets;
    this.input = input;
    this.sfx = sfx;

    this.hud = new HUDSystem();

    this.state = "title";
    this.globalFrame = 0;
    this.rooms = [];
    this.currentRoomIndex = 0;
    this.player = null;

    this.usedPreDialogs = new Set();
    this.currentDialog = null;
    this.dialogResult = null;
    this.dialogTimer = 0;

    this.messageTimer = 0;
    this.victory = false;
  }

  resetGame() {
    this.rooms = [];
    for (let i = 0; i < NUM_ROOMS; i++) {
      this.rooms.push(new Room(i, this.assets, (index) => this.playSfx(index), { isBoss: i === NUM_ROOMS - 1 }));
    }
    this.player = new PlayerShip(this.assets.ship, { playSfx: (index) => this.playSfx(index) });
    this.currentRoomIndex = 0;
    this.usedPreDialogs.clear();
    this.currentDialog = null;
    this.dialogResult = null;
    this.victory = false;
  }

  playSfx(index) {
    if (this.sfx) {
      this.sfx.play(index);
    }
  }

  startGame() {
    this.resetGame();
    this.state = "preDialog";
    this.currentDialog = createSequenceDialog(pickUnusedPreDialog(this.usedPreDialogs));
  }

  update() {
    this.globalFrame += 1;

    switch (this.state) {
      case "title":
        if (this.input.wasPressed("Enter") || this.input.wasPressed("Space")) {
          this.startGame();
        }
        break;
      case "preDialog":
        this.updatePreDialog();
        break;
      case "playing":
        this.updatePlaying();
        break;
      case "postDialog":
        this.updatePostDialog();
        break;
      case "gameOver":
        if (this.input.wasPressed("Enter") || this.input.wasPressed("Space")) {
          this.state = "title";
        }
        break;
    }
  }

  updatePreDialog() {
    if (!this.currentDialog) {
      this.state = "playing";
      return;
    }
    if (this.input.wasPressed("Enter") || this.input.wasPressed("Space")) {
      this.currentDialog.index += 1;
      if (this.currentDialog.index >= this.currentDialog.lines.length) {
        this.currentDialog = null;
        this.state = "playing";
      }
    }
  }

  updatePlaying() {
    const room = this.rooms[this.currentRoomIndex];
    this.player.update(this.input, room.enemies);
    room.update(this.player, this.globalFrame);

    if (this.player.hp <= 0) {
      this.state = "gameOver";
      this.victory = false;
      return;
    }

    if (room.isEnemyCleared()) {
      if (!room.collectionStarted) {
        room.startCollection();
      } else if (room.isCollectionDone()) {
        this.currentDialog = createChoiceDialog(postDialogs[Math.floor(Math.random() * postDialogs.length)]);
        this.state = "postDialog";
      }
    }
  }

  updatePostDialog() {
    const dialog = this.currentDialog;
    if (!dialog) {
      this.advanceRoom();
      return;
    }

    if (dialog.selection === null) {
      for (let i = 0; i < dialog.choices.length; i++) {
        const key = `Digit${i + 1}`;
        if (this.input.wasPressed(key)) {
          dialog.selection = i;
          applyDialogEffects(this.player, dialog.choices[i].effects);
          break;
        }
      }
    } else if (this.input.wasPressed("Enter") || this.input.wasPressed("Space")) {
      this.advanceRoom();
    }
  }

  advanceRoom() {
    this.currentRoomIndex += 1;
    if (this.currentRoomIndex >= this.rooms.length) {
      this.state = "gameOver";
      this.victory = true;
    } else {
      this.state = "preDialog";
      this.currentDialog = createSequenceDialog(pickUnusedPreDialog(this.usedPreDialogs));
    }
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    switch (this.state) {
      case "title":
        this.drawTitle();
        break;
      case "preDialog":
        this.drawRoom();
        this.drawPreDialog();
        break;
      case "playing":
        this.drawRoom();
        break;
      case "postDialog":
        this.drawRoom();
        this.drawPostDialog();
        break;
      case "gameOver":
        this.drawRoom();
        this.drawGameOver();
        break;
    }
  }

  drawRoom() {
    const room =
      this.rooms[this.currentRoomIndex] || this.rooms[this.rooms.length - 1];
    if (room) {
      room.draw(this.ctx);
    }
    if (this.player) {
      this.player.draw(this.ctx);
      this.hud.draw(this.ctx, this.player);
    }
  }

  drawTitle() {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = "#ffcc33";
    ctx.font = "bold 72px 'Orbitron', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("WEDGEROGUE", WIDTH / 2, HEIGHT / 2 - 50);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 28px sans-serif";
    const blink = Math.floor(this.globalFrame / 30) % 2 === 0;
    if (blink) {
      ctx.fillText("PRESS SPACE TO START", WIDTH / 2, HEIGHT / 2 + 40);
    }
    ctx.restore();
  }

  drawPreDialog() {
    if (!this.currentDialog) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(80, HEIGHT - 220, WIDTH - 160, 160);
    ctx.strokeStyle = "#ffcc33";
    ctx.lineWidth = 2;
    ctx.strokeRect(80, HEIGHT - 220, WIDTH - 160, 160);

    ctx.fillStyle = "#ffffff";
    ctx.font = "20px sans-serif";
    ctx.textAlign = "left";
    const text = this.currentDialog.lines[this.currentDialog.index] || "";
    const lines = text.split("\n");
    let y = HEIGHT - 180;
    for (const line of lines) {
      ctx.fillText(line, 110, y);
      y += 26;
    }

    ctx.fillStyle = "#888";
    ctx.font = "16px sans-serif";
    ctx.fillText("Press Space to continue", WIDTH - 240, HEIGHT - 70);
    ctx.restore();
  }

  drawPostDialog() {
    const ctx = this.ctx;
    const dialog = this.currentDialog;
    if (!dialog) return;

    ctx.save();
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(100, HEIGHT / 2 - 150, WIDTH - 200, 300);
    ctx.strokeStyle = "#00ffff";
    ctx.lineWidth = 2;
    ctx.strokeRect(100, HEIGHT / 2 - 150, WIDTH - 200, 300);

    ctx.fillStyle = "#ffffff";
    ctx.font = "20px sans-serif";
    ctx.textAlign = "left";
    const lines = dialog.prompt.split("\n");
    let y = HEIGHT / 2 - 110;
    for (const line of lines) {
      ctx.fillText(line, 130, y);
      y += 26;
    }

    y += 10;
    dialog.choices.forEach((choice, idx) => {
      const color = dialog.selection === idx ? "#ffcc33" : "#00ffff";
      ctx.fillStyle = color;
      ctx.fillText(choice.text, 130, y);
      y += 24;
      if (dialog.selection === idx) {
        ctx.fillStyle = "#aaaaaa";
        ctx.fillText(choice.result, 150, y);
        y += 28;
      }
    });

    if (dialog.selection !== null) {
      ctx.fillStyle = "#cccccc";
      ctx.font = "16px sans-serif";
      ctx.fillText("Press Enter to continue", WIDTH - 280, HEIGHT / 2 + 110);
    }

    ctx.restore();
  }

  drawGameOver() {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = this.victory ? "#8cff8c" : "#ff6666";
    ctx.font = "bold 64px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(this.victory ? "MISSION COMPLETE" : "SHIP DESTROYED", WIDTH / 2, HEIGHT / 2 - 20);

    ctx.fillStyle = "#ffffff";
    ctx.font = "24px sans-serif";
    ctx.fillText(`Final Score: ${Math.floor(this.player.score)}`, WIDTH / 2, HEIGHT / 2 + 40);
    ctx.fillText("Press Space to return to title", WIDTH / 2, HEIGHT / 2 + 80);
    ctx.restore();
  }
}
