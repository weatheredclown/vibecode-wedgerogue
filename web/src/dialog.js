export const preDialogPool = [
  [
    "COMMANDER TULIP:\nPilot, remember: strategy wins battles, not brute force.",
    "GENERAL XANTAR:\nBrute force works just fine if you use enough of it.",
    "DAN THE SPACE HAMSTER:\nI brute-forced my way into the snack stash yesterday. No regrets."
  ],
  [
    "TROY THE RHYTHM GURU:\nJust dodge to the beat, and you’ll be fine.",
    "COMMANDER TULIP:\nThis isn’t a dance recital, Troy.",
    "DAN THE SPACE HAMSTER:\nBut if it were, we’d definitely win 'best moves.'",
    "AI ASSISTANT:\nDancing detected. Enemies unimpressed. Continuing attack."
  ],
  [
    "GENERAL XANTAR:\nA true warrior must never hesitate.",
    "DAN THE SPACE HAMSTER:\nUnless there’s a cheese plate nearby. Priorities!",
    "COMMANDER TULIP:\nHamster, I’m starting to wonder how you made it onto this ship."
  ],
  [
    "AI ASSISTANT:\nLaser avoidance training is recommended.",
    "TROY THE RHYTHM GURU:\nOr just spin in circles while screaming. That works too.",
    "GENERAL XANTAR:\nThat’s a terrible strategy.",
    "DAN THE SPACE HAMSTER:\nIt’s more of a lifestyle, really."
  ],
  [
    "COMMANDER TULIP:\nYour mission is critical. Don’t mess it up.",
    "DAN THE SPACE HAMSTER:\nWow, motivational speeches are really your thing, huh?",
    "GENERAL XANTAR:\nMotivation isn’t required. Victory is.",
    "TROY THE RHYTHM GURU:\nI vote for snacks and good vibes instead. Who’s with me?",
    "AI ASSISTANT:\nSnacks do not improve survival rates."
  ],
  [
    "GENERAL XANTAR:\nThe enemies ahead are relentless. Prepare for a fight!",
    "COMMANDER TULIP:\nOr prepare to dodge. Whichever keeps us alive longer.",
    "AI ASSISTANT:\nDodge efficiency is currently suboptimal. Suggest improvement.",
    "TROY THE RHYTHM GURU:\nHey, suboptimal is my middle name!"
  ],
  [
    "TROY THE RHYTHM GURU:\nLife is a rhythm, pilot. Feel it, dodge it, win it.",
    "COMMANDER TULIP:\nI feel like your advice is always 90% nonsense.",
    "DAN THE SPACE HAMSTER:\nYeah, but the other 10% is gold. Like, actual gold sometimes."
  ],
  [
    "COMMANDER TULIP:\nThis ship is a finely tuned machine. Treat it with care.",
    "DAN THE SPACE HAMSTER:\nI thought it was held together by duct tape.",
    "AI ASSISTANT:\nStatement confirmed. Duct tape integrity at 84%.",
    "GENERAL XANTAR:\nIt’s enough for glory!"
  ],
  [
    "AI ASSISTANT:\nEnemy probability: 100%. Survival probability: questionable.",
    "DAN THE SPACE HAMSTER:\nI’m questioning why we brought you along.",
    "COMMANDER TULIP:\nShe’s saved us more times than you have, Hamster.",
    "TROY THE RHYTHM GURU:\nLet’s all agree to save each other this time."
  ],
  [
    "GENERAL XANTAR:\nA true warrior doesn’t fear death.",
    "DAN THE SPACE HAMSTER:\nI do! I fear it a lot!",
    "COMMANDER TULIP:\nThen channel that fear into something productive.",
    "TROY THE RHYTHM GURU:\nLike dodging! Fear-based dodging is the best kind."
  ],
  [
    "COMMANDER TULIP:\nStay focused, pilot. This is no time for distractions.",
    "DAN THE SPACE HAMSTER:\nUnless the distraction is snacks. Then it’s fine, right?",
    "AI ASSISTANT:\nSnacks detected: zero. Distraction detected: 100%.",
    "TROY THE RHYTHM GURU:\nDistraction? I call it 'creative improvisation.'",
    "GENERAL XANTAR:\nI call it 'a quick way to get us all vaporized.'"
  ],
  [
    "AI ASSISTANT:\nEnemies are numerous and heavily armed. Suggest caution.",
    "GENERAL XANTAR:\nA true warrior fears no enemy.",
    "DAN THE SPACE HAMSTER:\nWhat about highly explosive enemies? Asking for a friend.",
    "TROY THE RHYTHM GURU:\nExplosions are just fireworks you didn’t plan for!"
  ],
  [
    "COMMANDER TULIP:\nPilot, you’re the best we’ve got. Don’t let us down.",
    "DAN THE SPACE HAMSTER:\nWait, I thought *I* was the best we’ve got?",
    "GENERAL XANTAR:\nHamster, you’re barely on the list.",
    "TROY THE RHYTHM GURU:\nI’m on the list, right? For best vibes?"
  ],
  [
    "GENERAL XANTAR:\nThis room will test your limits, pilot.",
    "AI ASSISTANT:\nLimits detected: many. Breaking point estimated at 73%.",
    "TROY THE RHYTHM GURU:\nThat’s fine. Limits are just suggestions anyway.",
    "DAN THE SPACE HAMSTER:\nI suggest snacks. Lots of snacks."
  ],
  [
    "COMMANDER TULIP:\nEvery mission is a chance to prove yourself.",
    "DAN THE SPACE HAMSTER:\nProve myself? I’m already great!",
    "GENERAL XANTAR:\nGreat at eating everything in sight.",
    "TROY THE RHYTHM GURU:\nTo be fair, that’s an impressive skill.",
    "AI ASSISTANT:\nFood consumption rate: concerning. Recommend rationing."
  ]
];

export const postDialogs = [
  {
    prompt: "COMMANDER TULIP:\nWell done, pilot. What's next?",
    choices: [
      { text: "1) \"Onward to glory!\"", result: "Score +10", effects: { score: 10 } },
      { text: "2) \"Time for a snack break.\"", result: "Score +5", effects: { score: 5 } },
    ],
  },
  {
    prompt: "GENERAL XANTAR:\nShall I prepare you for the next battle?",
    choices: [
      { text: "1) \"Yes, full repairs!\"", result: "HP restored.", effects: { repair: true } },
      { text: "2) \"No, I’m ready as is.\"", result: "No changes.", effects: {} },
    ],
  },
];

export function pickUnusedPreDialog(used) {
  const available = preDialogPool
    .map((lines, idx) => ({ lines, idx }))
    .filter((entry) => !used.has(entry.idx));
  if (available.length === 0) {
    used.clear();
    const idx = Math.floor(Math.random() * preDialogPool.length);
    used.add(idx);
    return preDialogPool[idx];
  }
  const choice = available[Math.floor(Math.random() * available.length)];
  used.add(choice.idx);
  return choice.lines;
}

export function applyDialogEffects(player, effects) {
  if (!effects) return;
  if (typeof effects.score === "number") {
    player.score += effects.score;
  }
  if (effects.repair) {
    player.hp = player.maxHp;
  }
}

export function createSequenceDialog(lines) {
  return {
    type: "sequence",
    lines,
    index: 0,
  };
}

export function createChoiceDialog(dialog) {
  return {
    type: "choice",
    prompt: dialog.prompt,
    choices: dialog.choices,
    selection: null,
  };
}
