import pygame # For K_RETURN, K_1, K_2
import random
from utils import draw_text
from config import WIDTH, HEIGHT

# Data for dialogs
g_pre_conversations = [
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
]

g_post_dialogs = [
    {
        'prompt': "COMMANDER TULIP:\nWell done, pilot. What's next?",
        'choices': [
            ("1) \"Onward to glory!\"", "Score +10"),
            ("2) \"Time for a snack break.\"", "Score +5")
        ],
        'effects': [
            {'score': 10},
            {'score': 5}
        ]
    },
    {
        'prompt': "GENERAL XANTAR:\nShall I prepare you for the next battle?",
        'choices': [
            ("1) \"Yes, full repairs!\"", "HP restored."),
            ("2) \"No, I’m ready as is.\"", "No changes.")
        ],
        'effects': [
            {'repair': True},
            {}
        ]
    }
]

class DialogSystem:
    def __init__(self):
        self.pre_conversations = g_pre_conversations
        self.post_dialogs = g_post_dialogs
        self.used_pre_conversations = set()
        self.current_pre_conversation = []
        self.current_post_dialog = None
        self.state = 'idle'  # Can be 'idle', 'pre', 'post'
        self.timer = 0
        self.line_timer = 0
        self.DIALOG_TIME_PER_LINE = 240  # Frames per line (~4 seconds at 60 FPS)
        self.current_line = 0
        self.branch_mode = False  # True when showing choices in post-dialog
        self.selected_choice = None

    def is_pre_active(self):
        return self.state == 'pre'

    def is_post_active(self):
        return self.state == 'post'

    def start_post_dialog(self):
        if self.post_dialogs:
            self.current_post_dialog = random.choice(self.post_dialogs)
            self.branch_mode = True
            self.selected_choice = None
            self.state = 'post'
            self.timer = 0
        else:
            self.state = 'idle'

    def start_random_pre_dialog(self):
        available_conversations = [
            i for i in range(len(self.pre_conversations)) if i not in self.used_pre_conversations
        ]
        if available_conversations:
            selected_index = random.choice(available_conversations)
            self.used_pre_conversations.add(selected_index)
            self.current_pre_conversation = self.pre_conversations[selected_index]
            self.current_line = 0
            self.timer = 0
            self.line_timer = 0
            self.state = 'pre'
        else:
            # All pre-conversations used, perhaps reset or provide default
            self.current_pre_conversation = ["COMMANDER TULIP:\nWe've been through a lot, pilot. Impressive."]
            self.current_line = 0; self.timer = 0; self.line_timer = 0; self.state = 'pre'


    def next_line(self):
        self.current_line += 1
        self.line_timer = 0
        if self.current_line >= len(self.current_pre_conversation):
            self.state = 'idle'

    def update(self, keys, player): # Player object needed for applying effects
        if self.state == 'pre':
            self.line_timer += 1
            # Check for key press (pygame.K_RETURN) or if line_timer exceeds display time
            if (keys[pygame.K_RETURN] if pygame.K_RETURN < len(keys) else False) or \
               self.line_timer > self.DIALOG_TIME_PER_LINE:
                self.next_line()
        
        elif self.state == 'post' and self.current_post_dialog:
            if self.branch_mode:
                if (keys[pygame.K_1] if pygame.K_1 < len(keys) else False):
                    self.selected_choice = 0
                    self.branch_mode = False
                    self.apply_choice_effect(0, player)
                elif (keys[pygame.K_2] if pygame.K_2 < len(keys) else False) and \
                     len(self.current_post_dialog['choices']) > 1: # Ensure choice exists
                    self.selected_choice = 1
                    self.branch_mode = False
                    self.apply_choice_effect(1, player)
            else: # Waiting for timer to show result
                self.timer += 1
                if self.timer > self.DIALOG_TIME_PER_LINE: # Show result for a certain time
                    self.state = 'idle'

    def apply_choice_effect(self, choice_idx, player):
        if self.current_post_dialog and 0 <= choice_idx < len(self.current_post_dialog['effects']):
            effects = self.current_post_dialog['effects'][choice_idx]
            if player: # Ensure player object is valid
                if 'score' in effects:
                    player.score += effects['score']
                if 'repair' in effects and effects['repair']: # Assuming 'repair' is boolean
                    player.hp = player.max_hp # Example: Full repair

    def draw(self, surface):
        if self.state == 'pre' and self.current_pre_conversation:
            if self.current_line < len(self.current_pre_conversation):
                line = self.current_pre_conversation[self.current_line]
                draw_text(surface, line, WIDTH // 2, HEIGHT // 2 - 100, center=True)
        
        elif self.state == 'post' and self.current_post_dialog:
            prompt = self.current_post_dialog['prompt']
            draw_text(surface, prompt, WIDTH // 2, HEIGHT // 2 - 120, center=True)

            if self.branch_mode:
                y_offset = HEIGHT // 2 - 40
                for idx, (choice_text, _) in enumerate(self.current_post_dialog['choices']):
                    draw_text(surface, choice_text, WIDTH // 2, y_offset, color=(0, 255, 255), center=True)
                    y_offset += 40
            elif self.selected_choice is not None:
                # Display the consequence/description of the choice
                result_text = self.current_post_dialog['choices'][self.selected_choice][1]
                draw_text(surface, result_text, WIDTH // 2, HEIGHT // 2 - 40, center=True)

    def is_pre_done(self):
        return self.state != 'pre'

    def is_post_done(self):
        return self.state != 'post'
