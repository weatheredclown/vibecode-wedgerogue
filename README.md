# vibecode-wedgerogue
Vibe-Coding output based on "Create an Asteroids-Inspired Rogue-like with Colorful Characters and Branching Dialog" 

```
$ pip install pygame
$ python main.py
```

ai generated music, ai generated fm-synthesis soundfx, ai generated sprites

![screenshot 1](https://github.com/weatheredclown/vibecode-wedgerogue/blob/main/screenshots/wedgerogue01.png)
![screenshot 2](https://github.com/weatheredclown/vibecode-wedgerogue/blob/main/screenshots/wedgerogue02.png)

## JavaScript port

The `web/` directory contains a canvas-based port of the game. To try it
locally, serve the repository with any static file server and open
`web/index.html` in your browser:

```
npm install -g serve
serve .
```

Only the core combat loop is implemented today. The FM-synthesis audio engine,
shop/upgrade flow, options menu, and star map sequences from the Python version
are annotated with `TODO` comments for future work.

