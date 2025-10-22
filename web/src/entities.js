import {
  Vector2,
  wrapPosition,
  randRange,
  randInt,
  distanceSq,
  drawWrappedImage,
} from "./utils.js";
import {
  WIDTH,
  HEIGHT,
  ROTATION_SPEED,
  THRUST,
  MAX_SPEED,
  BULLET_INTERVAL,
} from "./constants.js";

const ENEMY_TYPES = [
  { color: [255, 0, 0], maxHealth: 5, points: 5 },
  { color: [255, 140, 0], maxHealth: 8, points: 10 },
  { color: [255, 255, 0], maxHealth: 12, points: 15 },
  { color: [0, 255, 0], maxHealth: 15, points: 20 },
  { color: [0, 200, 200], maxHealth: 20, points: 25 },
];

function rgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

export class Particle {
  constructor(pos, vel, color, shape = "circle", lifetime = 30, radius = 3) {
    this.pos = pos.clone();
    this.vel = vel.clone();
    this.color = color;
    this.shape = shape;
    this.lifetime = lifetime;
    this.maxLifetime = lifetime;
    this.radius = radius;
    this.timer = 0;
  }

  update() {
    this.pos.add(this.vel);
    this.lifetime -= 1;
    this.timer += 1;
  }

  draw(ctx) {
    const alphaRatio = Math.max(0, this.lifetime / this.maxLifetime);
    const [r, g, b] = this.color;
    ctx.save();
    ctx.globalAlpha = alphaRatio;
    ctx.fillStyle = rgb([r, g, b]);

    if (this.shape === "circle") {
      ctx.beginPath();
      ctx.arc(this.pos.x, this.pos.y, this.radius, 0, Math.PI * 2);
      ctx.fill();
    } else if (this.shape === "star") {
      const spikes = 5;
      const angleOffset = this.timer * 0.3;
      ctx.beginPath();
      for (let i = 0; i < spikes * 2; i++) {
        const radius = i % 2 === 0 ? this.radius : this.radius * 0.5;
        const angle = angleOffset + (Math.PI * 2 * i) / (spikes * 2);
        const x = this.pos.x + radius * Math.cos(angle);
        const y = this.pos.y + radius * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
    }

    ctx.restore();
  }

  isDead() {
    return this.lifetime <= 0;
  }
}

export class ParticleSystem {
  constructor() {
    this.particles = [];
  }

  spawnHitParticles(pos, color = [255, 200, 50], count = 8) {
    for (let i = 0; i < count; i++) {
      const angle = randRange(0, Math.PI * 2);
      const speed = randRange(2, 5);
      const vel = new Vector2(Math.cos(angle) * speed, Math.sin(angle) * speed);
      const particle = new Particle(pos, vel, color, "circle", 20, 3);
      this.particles.push(particle);
    }
  }

  spawnStarBurst(pos, color = [255, 255, 0], count = 8) {
    for (let i = 0; i < count; i++) {
      const angle = randRange(0, Math.PI * 2);
      const speed = randRange(1.5, 4.0);
      const vel = new Vector2(Math.cos(angle) * speed, Math.sin(angle) * speed);
      const particle = new Particle(pos, vel, color, "star", 30, 6);
      this.particles.push(particle);
    }
  }

  update() {
    for (const particle of this.particles) {
      particle.update();
    }
    this.particles = this.particles.filter((p) => !p.isDead());
  }

  draw(ctx) {
    for (const particle of this.particles) {
      particle.draw(ctx);
    }
  }
}

export class TrailSegment {
  constructor(pos, angle, color, lifetime = 30) {
    this.pos = pos.clone();
    this.angle = angle;
    this.color = color;
    this.lifetime = lifetime;
    this.maxLifetime = lifetime;
  }

  update() {
    this.lifetime -= 1;
  }

  draw(ctx) {
    const alphaRatio = Math.max(0, this.lifetime / this.maxLifetime);
    const [r, g, b] = this.color;
    ctx.save();
    ctx.globalAlpha = alphaRatio;
    ctx.fillStyle = rgb([r, g, b]);

    const basePoints = [
      new Vector2(15, 0),
      new Vector2(-8, 8),
      new Vector2(-8, -8),
    ];
    const rad = (this.angle * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    ctx.beginPath();
    basePoints.forEach((p, idx) => {
      const x = p.x * cos - p.y * sin + this.pos.x;
      const y = p.x * sin + p.y * cos + this.pos.y;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}

export class Bullet {
  constructor(pos, vel, { color = [255, 255, 255], homing = 0, bigger = false, exploding = false } = {}) {
    this.pos = pos.clone();
    this.vel = vel.clone();
    this.color = color;
    this.life = 120;
    this.homing = homing;
    this.bigger = bigger;
    this.exploding = exploding;
    this.radius = bigger ? 6 : 3;
  }

  update(enemies) {
    if (this.homing > 0 && enemies && enemies.length > 0) {
      let nearest = null;
      let bestDist = Infinity;
      for (const enemy of enemies) {
        const dist = distanceSq(enemy.pos, this.pos);
        if (dist < bestDist) {
          bestDist = dist;
          nearest = enemy;
        }
      }
      if (nearest) {
        const diff = nearest.pos.clone().sub(this.pos);
        const len = diff.length();
        if (len > 0) {
          diff.scale(1 / len);
          const currentSpeed = this.vel.length();
          const targetVel = diff.scale(currentSpeed);
          const steer = this.homing === 1 ? 0.05 : 0.1;
          this.vel.lerp(targetVel, steer);
        }
      }
    }
    this.pos.add(this.vel);
    this.life -= 1;
  }

  draw(ctx) {
    ctx.fillStyle = rgb(this.color);
    ctx.beginPath();
    ctx.arc(this.pos.x, this.pos.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
  }

  isDead() {
    return this.life <= 0;
  }
}

export class LootToken {
  constructor(pos, color, value = 10) {
    this.pos = pos.clone();
    this.vel = new Vector2(randRange(-1, 1), randRange(-1, 1));
    this.color = color;
    this.value = value;
    this.radius = 8;
  }

  update() {
    this.pos.add(this.vel.clone().scale(0.3));
    this.vel.scale(0.98);
    wrapPosition(this.pos, WIDTH, HEIGHT);
  }

  draw(ctx) {
    ctx.fillStyle = rgb(this.color);
    ctx.beginPath();
    ctx.arc(this.pos.x, this.pos.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
  }

  checkCollision(player) {
    const sum = this.radius + 20;
    return distanceSq(this.pos, player.pos) < sum * sum;
  }

  onCollected(particles) {
    particles.spawnStarBurst(this.pos, this.color, 10);
  }
}

export class FancyLootToken extends LootToken {
  constructor(pos, color, value, particles) {
    super(pos, color, value);
    this.timer = 0;
    this.particles = particles;
  }

  update() {
    this.timer += 1;
    const waveMag = 0.2;
    this.vel.y += waveMag * Math.sin(this.timer * 0.05);
    this.vel.x += waveMag * Math.cos(this.timer * 0.07);
    this.pos.add(this.vel.clone().scale(0.3));
    this.vel.scale(0.98);
  }

  draw(ctx) {
    const spikes = 5;
    const angle = this.timer * 0.1;
    ctx.fillStyle = rgb(this.color);
    ctx.beginPath();
    for (let i = 0; i < spikes * 2; i++) {
      const radius = i % 2 === 0 ? this.radius : this.radius * 0.5;
      const theta = angle + (Math.PI * 2 * i) / (spikes * 2);
      const x = this.pos.x + radius * Math.cos(theta);
      const y = this.pos.y + radius * Math.sin(theta);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }

  onCollected(particles) {
    super.onCollected(particles);
    if (this.particles) {
      this.particles.spawnStarBurst(this.pos, this.color, 10);
    }
  }
}

export class Enemy {
  constructor(pos, sprite, { playSfx, isBoss = false } = {}) {
    this.pos = pos.clone();
    this.vel = new Vector2(randRange(-2, 2), randRange(-2, 2));
    this.bullets = [];
    this.angle = 0;
    this.timer = 0;
    this.sprite = sprite;
    this.playSfx = playSfx;
    this.droppedLoot = false;

    const type = ENEMY_TYPES[randInt(0, ENEMY_TYPES.length - 1)];
    this.color = type.color;
    this.maxHealth = type.maxHealth;
    this.health = this.maxHealth;
    this.points = type.points;
    this.radius = 20;

    this.isBoss = isBoss;
    if (this.isBoss) {
      this.maxHealth = 200;
      this.health = this.maxHealth;
      this.radius = 30;
      this.color = [255, 80, 80];
    }
  }

  update(globalFrame, enemies) {
    this.pos.add(this.vel.clone().scale(0.5));
    wrapPosition(this.pos, WIDTH, HEIGHT);
    this.timer += 1;

    if (this.timer % BULLET_INTERVAL === 0) {
      const step = this.isBoss ? 15 : 30;
      for (let angleDeg = 0; angleDeg < 360; angleDeg += step) {
        const angleRad = ((angleDeg + this.timer * 5) * Math.PI) / 180;
        const bx = Math.cos(angleRad);
        const by = Math.sin(angleRad);
        const speed = 3 + Math.sin(globalFrame / 20);
        if (this.playSfx) this.playSfx(10);
        this.bullets.push(
          new Bullet(this.pos, new Vector2(bx * speed, by * speed), {
            color: this.color,
          })
        );
      }
    }

    for (const bullet of this.bullets) {
      bullet.update();
    }
    this.bullets = this.bullets.filter((b) => !b.isDead());
  }

  draw(ctx) {
    // diamond outline
    const diamond = this.getDiamondPoints();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    diamond.forEach(([x, y], idx) => {
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();

    const fraction = Math.max(0, Math.min(1, this.health / this.maxHealth));
    if (fraction > 0) {
      const fillPoly = this.getHealthFillPolygon(fraction);
      ctx.fillStyle = rgb(this.color);
      ctx.beginPath();
      fillPoly.forEach(([x, y], idx) => {
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
    }

    for (const bullet of this.bullets) {
      bullet.draw(ctx);
    }

    if (this.sprite) {
      drawWrappedImage(ctx, this.sprite, this.pos.x, this.pos.y, -this.angle);
    }
  }

  getDiamondPoints() {
    const { x, y } = this.pos;
    const r = this.radius;
    return [
      [x, y - r],
      [x + r, y],
      [x, y + r],
      [x - r, y],
    ];
  }

  getHealthFillPolygon(fraction) {
    const diamond = this.getDiamondPoints();
    const top = diamond[0];
    const right = diamond[1];
    const bottom = diamond[2];
    const left = diamond[3];

    const fillY = bottom[1] + fraction * (top[1] - bottom[1]);
    const edges = [
      [top, right],
      [right, bottom],
      [bottom, left],
      [left, top],
    ];

    const points = [];
    for (const [p1, p2] of edges) {
      if (p1[1] >= fillY) {
        points.push(p1);
      }
      const cross = this.edgeIntersection(p1, p2, fillY);
      if (cross) points.push(cross);
    }

    const cx = this.pos.x;
    const cy = this.pos.y;
    const unique = [];
    for (const pt of points) {
      if (!unique.some((other) => Math.abs(other[0] - pt[0]) < 0.01 && Math.abs(other[1] - pt[1]) < 0.01)) {
        unique.push(pt);
      }
    }
    unique.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
    return unique;
  }

  edgeIntersection(p1, p2, lineY) {
    const [x1, y1] = p1;
    const [x2, y2] = p2;
    if ((y1 < lineY && y2 < lineY) || (y1 > lineY && y2 > lineY)) return null;
    if (Math.abs(y2 - y1) < 1e-6) return null;
    const t = (lineY - y1) / (y2 - y1);
    if (t < 0 || t > 1) return null;
    const x = x1 + t * (x2 - x1);
    return [x, lineY];
  }

  hitBy(bullet) {
    const radius = this.radius + bullet.radius;
    return distanceSq(this.pos, bullet.pos) < radius * radius;
  }
}

export class PlayerShip {
  constructor(sprite, { playSfx }) {
    this.pos = new Vector2(WIDTH / 2, HEIGHT / 2);
    this.vel = new Vector2(0, 0);
    this.angle = -90;
    this.color = [255, 255, 255];
    this.bullets = [];
    this.trails = [];
    this.cooldown = 0;
    this.score = 0;
    this.engineTick = 0;

    this.maxHp = 50;
    this.hp = 50;
    this.shieldHp = 0;
    this.shieldMax = 0;

    this.upgrades = {
      homing_bullet_level: 0,
      bigger_bullets: false,
      exploding_bullets: false,
      shield: false,
      teleporter: false,
      short_range_autofire: false,
      mines: false,
      homing_mines: false,
    };

    this.altMode = true;
    this.sprite = sprite;
    this.playSfx = playSfx;
  }

  draw(ctx) {
    for (const trail of this.trails) {
      trail.draw(ctx);
    }

    const basePoints = [
      new Vector2(20, 0),
      new Vector2(-10, 10),
      new Vector2(-10, -10),
    ];
    const rad = (this.angle * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    ctx.strokeStyle = rgb(this.color);
    ctx.lineWidth = 2;
    ctx.beginPath();
    basePoints.forEach((p, idx) => {
      const x = p.x * cos - p.y * sin + this.pos.x;
      const y = p.x * sin + p.y * cos + this.pos.y;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();

    for (const bullet of this.bullets) {
      bullet.draw(ctx);
    }

    if (this.sprite) {
      drawWrappedImage(ctx, this.sprite, this.pos.x, this.pos.y, -this.angle);
    }
  }

  update(input, enemies) {
    if (input.wasPressed("KeyP")) {
      this.altMode = !this.altMode;
    }

    if (!this.altMode) {
      if (input.isDown("ArrowLeft")) this.angle -= ROTATION_SPEED;
      if (input.isDown("ArrowRight")) this.angle += ROTATION_SPEED;
      if (input.isDown("ArrowUp")) {
        const rad = (this.angle * Math.PI) / 180;
        const force = new Vector2(Math.cos(rad), Math.sin(rad)).scale(THRUST);
        this.vel.add(force);
        if (this.vel.length() > MAX_SPEED) this.vel.scaleToLength(MAX_SPEED);
      }
      if (input.isDown("ArrowDown")) {
        const rad = (this.angle * Math.PI) / 180;
        const force = new Vector2(Math.cos(rad), Math.sin(rad)).scale(-THRUST * 0.5);
        this.vel.add(force);
        if (this.vel.length() > MAX_SPEED) this.vel.scaleToLength(MAX_SPEED);
      }
    } else {
      const accel = new Vector2(0, 0);
      const accelAmount = 3;
      const friction = 0.8;
      const rate = 4;

      if (input.isDown("ArrowLeft")) {
        accel.x -= accelAmount;
        this.maybePlayEngine(rate, () => this.playSfx && this.playSfx(22));
      }
      if (input.isDown("ArrowRight")) {
        accel.x += accelAmount;
        this.maybePlayEngine(rate, () => this.playSfx && this.playSfx(20));
      }
      if (input.isDown("ArrowUp")) {
        accel.y -= accelAmount;
        this.maybePlayEngine(rate, () => this.playSfx && this.playSfx(19));
      }
      if (input.isDown("ArrowDown")) {
        accel.y += accelAmount;
        this.maybePlayEngine(rate, () => this.playSfx && this.playSfx(18));
      }

      this.vel.add(accel);
      if (this.vel.length() > MAX_SPEED) this.vel.scaleToLength(MAX_SPEED);
      if (accel.lengthSq() < 1e-6) this.vel.scale(friction);
      const speed = this.vel.length();
      if (speed > 0.1) {
        this.angle = (Math.atan2(this.vel.y, this.vel.x) * 180) / Math.PI;
      }
    }

    if (this.upgrades.teleporter && input.wasPressed("KeyT")) {
      if (this.playSfx) this.playSfx(1);
      this.pos.set(randInt(0, WIDTH), randInt(0, HEIGHT));
    }

    if (this.upgrades.mines && input.wasPressed("KeyM")) {
      if (this.playSfx) this.playSfx(2);
      this.dropMine();
    }

    if (input.isDown("Space")) {
      this.shoot();
    }

    if (this.upgrades.short_range_autofire) {
      this.autoFire(enemies);
    }

    this.pos.add(this.vel);
    wrapPosition(this.pos, WIDTH, HEIGHT);

    for (const bullet of this.bullets) {
      const targetEnemies = bullet.homing > 0 ? enemies : null;
      bullet.update(targetEnemies);
    }
    this.bullets = this.bullets.filter((b) => !b.isDead());

    for (const trail of this.trails) {
      trail.update();
    }
    this.trails = this.trails.filter((t) => t.lifetime > 0);
    this.trails.push(new TrailSegment(this.pos, this.angle, [0, 255, 0], 30));

    if (this.cooldown > 0) this.cooldown -= 1;
  }

  maybePlayEngine(rate, playFn) {
    this.engineTick = (this.engineTick + 1) % rate;
    if (this.engineTick === 0) {
      playFn();
    }
  }

  shoot() {
    if (this.cooldown > 0) return;
    const rad = (this.angle * Math.PI) / 180;
    const dir = new Vector2(Math.cos(rad), Math.sin(rad));
    const speed = 10;
    const bullet = new Bullet(this.pos.clone().add(dir.clone().scale(20)), dir.clone().scale(speed), {
      color: [0, 255, 0],
      homing: this.upgrades.homing_bullet_level,
      bigger: this.upgrades.bigger_bullets,
      exploding: this.upgrades.exploding_bullets,
    });
    this.bullets.push(bullet);
    this.cooldown = 10;
    if (this.playSfx) this.playSfx(10);
  }

  dropMine() {
    const vel = this.upgrades.homing_mines
      ? new Vector2(randRange(-1, 1), randRange(-1, 1))
      : new Vector2(0, 0);
    const bullet = new Bullet(this.pos.clone(), vel, {
      color: [255, 0, 0],
      homing: this.upgrades.homing_mines ? 2 : 0,
      bigger: true,
      exploding: true,
    });
    bullet.life = 300;
    this.bullets.push(bullet);
  }

  autoFire(enemies) {
    if (!enemies || enemies.length === 0) return;
    for (const enemy of enemies) {
      const dist = Math.sqrt(distanceSq(enemy.pos, this.pos));
      if (dist < 200 && this.cooldown <= 0) {
        const diff = enemy.pos.clone().sub(this.pos);
        this.angle = (Math.atan2(diff.y, diff.x) * 180) / Math.PI;
        if (this.playSfx) this.playSfx(3);
        this.shoot();
        break;
      }
    }
  }
}
