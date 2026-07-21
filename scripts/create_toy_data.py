"""Create synthetic toy data for running the entire MiniLLM pipeline offline.

Generates:
  - data/raw/toy_train.txt       (~200 lines of children's stories)
  - data/raw/toy_valid.txt       (~50 lines of children's stories)
  - data/sft/train.jsonl         (50 instruction-response pairs)
  - data/sft/valid.jsonl         (10 instruction-response pairs)
  - data/dpo/train.jsonl         (30 chosen/rejected pairs)
  - data/dpo/valid.jsonl         (10 chosen/rejected pairs)
  - data/prompts/eval_prompts.jsonl  (20 evaluation prompts, 5 types)
"""

import argparse
import json
import os
import random

# ---------------------------------------------------------------------------
# Story corpus -- simple English sentences about animals, nature, daily life
# ---------------------------------------------------------------------------

TRAIN_STORIES = [
    "Once upon a time, a little cat named Whiskers lived in a cozy house by the river. Every morning Whiskers would walk to the garden and chase butterflies among the flowers. One day she found a bright red ball hidden under a large green leaf. She batted the ball with her paw and it rolled all the way to the fence. Whiskers ran after it and giggled as the ball bounced over a small stone. By sunset she was tired and curled up on the soft rug by the fireplace.",
    "A brave little dog called Benny loved to explore the forest behind his home. One sunny afternoon he trotted down a narrow trail lined with tall pine trees. He heard birds singing sweetly in the branches above him. Suddenly he saw a small rabbit sitting next to a fallen log. Benny wagged his tail and the rabbit hopped closer. They became friends and played hide-and-seek among the bushes until the sky turned orange.",
    "In a quiet pond at the edge of the village lived a cheerful frog named Fred. Fred had bright green skin and big round eyes. Every evening he would sit on a lily pad and sing songs to the stars. The fish in the pond loved his music and swam in circles when he sang. One night a kind owl flew down and told Fred that his voice was the most beautiful in the whole forest.",
    "Emma and her brother Liam went to the park on a warm spring day. The grass was soft and green and colorful flowers bloomed everywhere. Emma brought her favorite blue kite and ran across the field to make it fly. The kite soared high into the sky dancing among the white clouds. Liam chased after her laughing and clapping his hands with joy.",
    "A tiny mouse named Pip lived inside the walls of an old bakery. Every night when the baker went home Pip would sneak out to find crumbs. He loved the sweet smell of fresh bread and sugary cookies. One evening he met another mouse called Dot who showed him a secret tunnel to the cheese shop next door. From that day on Pip and Dot shared the best treats in town.",
    "The sun rose over a beautiful green meadow where a family of deer grazed peacefully. The mother deer watched her two fawns as they chased each other through the tall grass. Butterflies floated past and bees buzzed around the wildflowers. A gentle breeze carried the scent of honeysuckle across the field. It was a perfect morning in the countryside.",
    "Oliver the owl was the wisest bird in the forest. Every night he sat on his favorite branch and watched over the woodland creatures. When a young squirrel lost his acorns Oliver helped him find them. When the rabbits argued over a burrow Oliver settled the dispute fairly. All the animals respected Oliver and came to him for advice.",
    "A little girl named Mia planted a sunflower seed in her backyard. She watered it every morning and watched as a tiny green shoot appeared. Day by day the shoot grew taller and stronger. By summer the sunflower was taller than Mia herself. Its golden petals opened toward the sun and smiled down at her. Mia was so proud of her beautiful sunflower.",
    "At the edge of town there was a small library run by a kind woman named Mrs. Rose. Children from all over the neighborhood came to read books and listen to stories. Every Saturday Mrs. Rose read a new adventure aloud. The children sat in a circle on colorful cushions their eyes wide with wonder. The library was the happiest place in the whole town.",
    "A playful dolphin named Splash lived in the deep blue ocean. He loved jumping out of the water and spinning in the air. One day he met a shy sea turtle named Shelley who was afraid of the waves. Splash taught Shelley how to ride the gentle currents and soon she was gliding through the water with a big smile on her face.",
    "Jack and his father went fishing at the lake early one Saturday morning. The water was calm and reflected the pink dawn sky like a mirror. Jack baited his hook and cast the line into the water. They waited quietly listening to the birds and watching the dragonflies. After a while Jack felt a tug on his line and pulled up a shiny silver fish. His father cheered and gave him a big high five.",
    "In a garden behind a red brick house a family of ladybugs made their home on a rose bush. The smallest ladybug was named Ruby and she had exactly seven spots on her wings. Every morning Ruby would crawl to the top of the tallest rose and look at the world below. She dreamed of flying over the garden wall to see what lay beyond. One sunny day a gust of wind lifted her up and she soared higher than she had ever gone before.",
    "A friendly bear named Bruno lived in a cave near the mountain top. Every autumn he gathered berries and honey to prepare for winter. The other animals often stopped by to chat and share news. Bruno always had a warm smile and a pot of tea ready for visitors. When the snow fell he curled up in his cozy cave and dreamed of spring.",
    "Lily loved to paint pictures of the ocean. She would sit on the sandy beach with her easel and brushes and capture the waves on canvas. Seagulls flew overhead and the salty breeze tickled her face. One day a curious seal popped its head out of the water and watched her paint. Lily laughed and included the seal in her masterpiece.",
    "A colony of ants worked together to build a magnificent anthill. Each ant had a special job to do. Some gathered food while others dug tunnels. The youngest ant named Andy was in charge of carrying tiny pebbles. Even though the work was hard Andy felt proud to help his family. Together the ants built one of the biggest anthills the garden had ever seen.",
    "Sophie and her grandmother baked chocolate chip cookies on a rainy afternoon. The kitchen smelled wonderful as the cookies baked in the oven. Sophie stirred the dough while her grandmother measured the flour. When the cookies were ready they sat by the window and ate them with glasses of cold milk. The rain outside made everything feel warm and cozy inside.",
    "A clever fox named Felix lived at the edge of the enchanted forest. He had bright orange fur and a bushy tail that shimmered in the sunlight. Felix loved solving riddles and puzzles that the forest creatures gave him. One day a crow challenged him to guess the color of the wind. Felix thought carefully and answered that the wind is the color of whatever it touches. The crow was impressed and declared Felix the smartest animal in the forest.",
    "At the county fair a little pony named Sparkles gave rides to children. She had a white coat with gray spots and a mane that flowed like silk. Children lined up eagerly to ride her around the ring. Sparkles loved making the children smile and always walked at a gentle pace. At the end of the day the fair workers gave her a special apple as a reward for being the best pony.",
    "A young robin named Red built his first nest in a tall oak tree. He gathered twigs and soft moss and wove them together carefully. It took him three whole days to finish. When the nest was complete he sat inside and looked out at the world feeling very proud. Soon a female robin came by and admired his work. They decided to share the nest and raise a family together.",
    "The children at Sunny Day School loved art class with Mr. Chen. He taught them how to mix colors and paint beautiful landscapes. One week they painted mountains with snow on top. The next week they painted oceans filled with colorful fish. Mr. Chen always said that every child is an artist. The hallway outside the classroom was covered with their bright cheerful paintings.",
    "A magical garden appeared in the center of town one spring morning. The flowers glowed with soft pastel light and the trees hummed gentle melodies. People came from all around to see the wonderful garden. A little boy named Tom discovered that if he whispered a wish to a blue flower the wish would come true by sunset. He wished for a puppy and that evening found a golden retriever waiting at his door.",
    "Grandpa and his granddaughter Zoe walked through the farmer's market on a crisp autumn day. They bought red apples and golden pumpkins and jars of honey. Zoe picked out the biggest pumpkin she could find and Grandpa helped her carry it. When they got home they carved a happy face into the pumpkin and placed a candle inside. It glowed warmly on the porch all through the evening.",
    "A tiny hummingbird named Zoom was the fastest flyer in the garden. Her wings beat so quickly they made a humming sound like a tiny motor. Zoom loved to dart from flower to flower sipping sweet nectar. One day she raced a bumblebee from one end of the garden to the other. Zoom won easily but she stopped to share a sip of nectar with the bee because she was kind as well as fast.",
    "In a small village by the sea the fishermen went out every morning before dawn. Their boats bobbed gently on the dark waves as they cast their nets into the deep water. By midday they returned with nets full of silver fish. The villagers gathered at the dock to buy the fresh catch. Everyone agreed that the fish from their little village were the tastiest in all the land.",
    "A curious kitten named Patches discovered a secret door behind the bookshelf in the living room. She squeezed through the opening and found a hidden room full of dusty old books and strange objects. There was a globe that showed countries she had never heard of and a clock that ran backwards. Patches explored every corner of the room and decided it would be her special adventure place.",
    "The stars in the night sky held a grand dance every evening when the world was asleep. They twinkled and spun in patterns that told stories of long ago. A shooting star named Stella was the best dancer of all. She would zoom across the sky leaving a trail of sparkling light behind her. The moon watched over the dance and smiled at the joy below.",
    "A kind farmer named Mr. Brown grew the biggest vegetables in the county. His carrots were as long as a ruler and his tomatoes were the size of softballs. Every year he entered the harvest festival and won first prize. But what made Mr. Brown happiest was sharing his vegetables with his neighbors. He said that food tastes best when it is shared with friends.",
    "Maya and her best friend Alex built a treehouse in the old maple tree in Maya's backyard. They hammered boards together and painted the walls blue and yellow. They put up a rope ladder and hung a sign that said Friends Only. Every afternoon after school they climbed up to their treehouse and told each other stories and made plans for new adventures.",
    "A sleepy sloth named Sam lived in a towering kapok tree in the rainforest. He spent most of his day hanging upside down and munching on leaves. The other animals thought Sam was lazy but he was actually very thoughtful. While hanging in his tree he noticed things that no one else did. He saw a rare blue butterfly that scientists had been searching for and helped them find it by pointing with his long claw.",
    "The river that ran through the valley was home to many creatures. Fish swam in its clear waters and otters played on its banks. Dragonflies hovered above the surface and frogs sat on rocks singing their songs. One spring the river swelled with rain and the animals worked together to build a dam of sticks and stones to protect their homes. The river taught them that cooperation makes everything possible.",
    "A young wizard named Eli found a dusty spell book in his attic. The book was old and its pages were yellowed with age. Eli read the first spell aloud and suddenly a small flame appeared in his palm. He practiced every day learning new spells and getting better each time. His favorite spell turned stones into butterflies that fluttered around his room in brilliant colors.",
    "Hannah loved to watch the clouds on lazy summer afternoons. She would lie on the grass and imagine shapes in the fluffy white clouds above. One day she saw a cloud that looked exactly like a castle with tall towers. Another cloud looked like a dragon breathing fire. She told her little brother about the cloud stories and he loved them so much that they made a game of it every weekend.",
    "A brave little tugboat named Tuggy worked in the busy harbor. Even though he was the smallest boat he had the biggest heart. When a giant cargo ship lost its engine Tuggy tied a rope to its bow and pulled it safely to the dock. Everyone cheered for the little tugboat. Tuggy learned that it does not matter how big you are as long as you have courage and a kind spirit.",
    "The school playground was the favorite place for all the children at Elm Street Elementary. There were swings that went high into the air and a slide that twisted like a corkscrew. A group of friends invented a game where they pretended the playground was a pirate ship sailing across a vast ocean. The monkey bars were the rigging and the sandbox was a desert island. Every recess was a new adventure.",
    "A pair of swans named Grace and Noble lived on a peaceful lake surrounded by willow trees. Every spring they built a nest of reeds and straw near the water's edge. They took turns sitting on their eggs keeping them warm and safe. When the cygnets hatched they paddled behind their parents in a perfect line. The family glided across the lake like a beautiful painting come to life.",
    "Leo the lizard loved to sunbathe on a warm flat rock in the desert. His skin changed color depending on his mood. When he was happy he turned bright green. When he was surprised he turned pale yellow. One day a hawk flew overhead and Leo quickly turned the color of the rock to hide. The hawk flew away without seeing him. Leo smiled knowing that being yourself and being clever could save the day.",
    "Mrs. Patel's kindergarten class had a pet hamster named Cinnamon. Every morning the children would take turns feeding Cinnamon and giving her fresh water. They loved watching her run on her little wheel and stuff her cheeks with sunflower seeds. On Fridays Mrs. Patel let a different student take Cinnamon home for the weekend. Taking care of a pet taught the children responsibility and kindness.",
    "A beautiful cherry blossom tree stood in the center of the town square. Every spring it burst into clouds of pink flowers that drifted like snow in the breeze. The townspeople held a festival under the tree with music and dancing and sweet treats. Children caught petals in their hands and made wishes. The cherry blossom tree was a symbol of hope and new beginnings for everyone in the town.",
    "Deep in the ocean a young whale named Walter sang songs that could be heard for miles. His mother taught him the ancient melodies that whales had sung for thousands of years. Walter practiced every day adding his own little twists to the traditional tunes. The other sea creatures would stop and listen whenever Walter sang. His music brought peace to the ocean and joy to every creature who heard it.",
    "A clever raccoon named Rocket lived in the attic of an old movie theater. At night when the building was empty Rocket would sneak down to the projection room and watch old films on the big screen. He learned about faraway places and exciting adventures. Rocket especially loved cowboy movies and would try to lasso pieces of popcorn with his tail. The theater owner discovered Rocket one night and instead of being angry he left out a bowl of popcorn every evening.",
    "One snowy morning the children woke up to find the world covered in white. They put on their warm coats and boots and ran outside to play. They built a snowman with a carrot nose and a scarf made of red wool. They had a snowball fight and made snow angels on the lawn. When their fingers got cold they went inside and drank hot chocolate with marshmallows floating on top.",
    "A tiny seed was carried by the wind across a vast desert. It landed in a small patch of dirt beside a lonely rock. The seed waited patiently for rain. When a gentle shower finally came the seed began to grow. A small green shoot pushed through the soil and reached toward the sun. Weeks later a beautiful desert rose bloomed beside the rock. The rock was no longer lonely because now it had a friend.",
    "The town baker Monsieur Dupont made the most delicious croissants in all of France. Every morning people lined up outside his shop before the sun even rose. He used the finest butter and let the dough rest for exactly twenty four hours. His secret ingredient was a pinch of love in every batch. When you bit into one of his croissants the flaky layers melted in your mouth like a warm hug.",
    "A little spider named Silvia spun the most beautiful webs in the garden. Each morning she would start with a single thread and work outward in perfect spirals. The dew drops on her web sparkled like diamonds in the morning light. The garden insects marveled at her artistry. Silvia told them that patience and practice were the secrets to her beautiful webs.",
    "At the top of the highest hill there was an old observatory where a friendly astronomer named Dr. Star lived. Every clear night she would open the telescope dome and gaze at the planets and stars. Children from the town would visit and look through the telescope at the rings of Saturn and the craters on the moon. Dr. Star taught them that each star was a sun with its own story to tell.",
    "A young penguin named Percy was afraid of the cold ocean water. While the other penguins dived and swam and caught fish Percy stood on the ice watching nervously. His mother told him that being brave does not mean not being afraid. It means trying even when you are scared. Percy took a deep breath and jumped into the water. It was cold at first but then he discovered how wonderful it felt to swim and play with his friends.",
    "The village clock tower chimed every hour on the hour without fail. Old Mr. Thompson the clock keeper wound the gears every Sunday morning. He had been doing this for forty years. The villagers set their watches by the tower and children counted the chimes to tell time. One day a storm damaged the clock and the town fell silent. Mr. Thompson worked through the night to repair it. When the clock chimed again the next morning the whole village cheered.",
    "A talented young girl named Rosa played the violin in the city orchestra. She practiced every day after school playing scales and sonatas until her fingers were sore. Her violin sang with a voice that could make people laugh or cry. One evening she played a solo in front of a thousand people. When she finished the audience was silent for a moment and then burst into thunderous applause. Rosa knew then that all her hard work had been worth it.",
    "In a flower shop on Main Street a cat named Blossom had a very special job. She sat in the window display among the roses and lilies and attracted customers with her calm friendly presence. People would stop to pet Blossom and end up buying a bouquet. The shop owner Mrs. Kim said that Blossom was her best employee. Every evening Blossom got a dish of cream as her salary.",
    "A family of ducks lived on a quiet pond near the farmhouse. Every morning the mother duck led her ducklings in a single file line to the water. They paddled around the pond looking for bread crumbs and tiny fish. The farmer's children loved to watch the ducks and would bring them corn to eat. The ducks quacked happily knowing they were safe and loved on the little farm.",
]

VALID_STORIES = [
    "A small turtle named Tommy decided to climb the biggest hill in the meadow. All the other animals told him he was too slow. But Tommy kept walking one step at a time. By afternoon he reached the top and could see the whole valley spread out below him. The view was breathtaking. Tommy smiled and said that slow and steady always wins the race.",
    "The school garden was a special project that every class helped maintain. The first graders planted carrots and the third graders grew tomatoes. The fifth graders were in charge of the sunflowers. Every morning before class the students would water their plants and pull out weeds. When harvest time came they had a big salad party and everyone agreed that vegetables taste better when you grow them yourself.",
    "A little girl named Abby found a starfish on the beach one morning. It was still alive but stranded on the sand. Abby carefully picked it up and walked it back to the ocean. She watched as the waves carried it away. The next day Abby found five more starfish and helped each one back into the water. She knew she could not save every starfish but she could make a difference for the ones she found.",
    "The autumn leaves fell in brilliant shades of red and orange and gold. A young boy named Marcus collected the most beautiful leaves and pressed them in a heavy book. He made cards and bookmarks for his family with the pressed leaves. His grandmother said they were the most beautiful gifts she had ever received. Marcus decided that autumn was his favorite season because it turned the world into art.",
    "A family of rabbits lived under an old wooden shed in the backyard. The baby bunnies were fluffy and white with pink noses and long ears. Every evening they hopped out to explore the yard. They nibbled on clover and chased each other around the garden gnome. The family dog watched them from the porch but never bothered them. The rabbits knew they were safe and happy in their cozy little home.",
    "The ice cream truck arrived on Maple Street every summer afternoon at exactly three o'clock. Children would run out of their houses with coins jingling in their hands. The truck played a cheerful tune that could be heard from blocks away. Mr. Garcia the ice cream man knew every child's favorite flavor. He always had a smile and an extra sprinkle for good measure.",
    "A young girl named Priya loved to gaze at the moon through her bedroom window. She kept a moon journal and drew its shape every night. She watched it change from a thin crescent to a full bright circle and back again. Her father told her that people all over the world saw the same moon. Priya felt connected to everyone on Earth when she looked up at the sky.",
    "The old oak tree in the schoolyard was over a hundred years old. Generations of children had climbed its sturdy branches and carved their initials into its bark. The tree provided shade on hot days and dropped acorns in the autumn. When the school wanted to cut it down to build a new gymnasium the children wrote letters and collected signatures to save it. The oak tree still stands today a living monument to the power of standing up for what you love.",
    "A happy little cloud named Fluffy floated across the sky looking for a place to rain. He passed over the ocean but the ocean already had plenty of water. He passed over the rainforest but the trees there were already green and lush. Finally he found a dry brown field where nothing was growing. Fluffy squeezed out all his rain and the field turned green with new life. Fluffy felt proud that he had helped something beautiful grow.",
    "Every winter a kind woman named Mrs. Chen set up a bird feeder in her front yard. She filled it with seeds and suet and watched as chickadees and cardinals and blue jays came to eat. She kept a bird identification book by the window and checked off every species she saw. The birds learned to trust her and would sing outside her window every morning as if to say thank you for the food.",
]

# ---------------------------------------------------------------------------
# SFT pair generators
# ---------------------------------------------------------------------------

_TOPICS = [
    "a brave cat", "a friendly dog", "a little bird", "a wise owl",
    "a happy frog", "a curious kitten", "a playful dolphin", "a tiny mouse",
    "a sleepy bear", "a kind rabbit", "a colorful butterfly", "a fast horse",
    "a gentle deer", "a clever fox", "a shy turtle", "a singing whale",
    "a busy ant", "a bright star", "a magic garden", "a cozy house",
]

_EMOTIONS = ["happy", "sad", "funny", "exciting", "surprising"]

_CONTINUATION_INSTRUCTIONS = [
    "Continue the story: Once upon a time there was a small village",
    "Continue the story: The sun set behind the mountains and",
    "Continue the story: A little girl opened the mysterious door and",
    "Continue the story: The animals gathered in the forest clearing to",
    "Continue the story: It was the first day of spring and",
    "Continue the story: The old lighthouse on the cliff began to",
    "Continue the story: A golden key was hidden under the old oak tree",
    "Continue the story: The river flowed gently through the valley where",
    "Continue the story: The children discovered a trail of sparkling footprints",
    "Continue the story: High above the clouds a tiny airplane",
]


def _make_sft_pairs(stories, num_samples, seed=42):
    """Generate SFT instruction-response pairs from stories."""
    rng = random.Random(seed)
    pairs = []

    # Strategy 1: "Write a story about X" -> full story
    for i in range(min(num_samples // 2, len(stories))):
        topic = rng.choice(_TOPICS)
        prompt = f"Write a short story about {topic}."
        response = stories[i % len(stories)]
        pairs.append({"prompt": prompt, "response": response})

    # Strategy 2: Continuation prompts from stories
    for i in range(min(num_samples // 2, len(stories))):
        story = stories[(i + len(stories) // 2) % len(stories)]
        sentences = story.replace("! ", ". ").replace("? ", ". ").split(". ")
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            split_point = min(2, len(sentences) - 1)
            instruction = "Continue the story: " + ". ".join(sentences[:split_point]) + "."
            response = ". ".join(sentences[split_point:])
            if not response.endswith("."):
                response += "."
            pairs.append({"prompt": instruction, "response": response})

    # Strategy 3: Topic + emotion style
    for i in range(num_samples - len(pairs)):
        topic = rng.choice(_TOPICS)
        emotion = rng.choice(_EMOTIONS)
        prompt = f"Write a {emotion} short story about {topic}."
        response = stories[(i + 3) % len(stories)]
        pairs.append({"prompt": prompt, "response": response})

    return pairs[:num_samples]


# ---------------------------------------------------------------------------
# DPO pair generators
# ---------------------------------------------------------------------------

def _corrupt_rejection(response: str, rng: random.Random) -> str:
    """Create a rejected response by corrupting the chosen one."""
    sentences = response.replace("! ", ". ").replace("? ", ". ").split(". ")
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return response

    method = rng.randint(0, 2)
    if method == 0:
        # Truncation: keep only first sentence or two
        keep = max(1, len(sentences) // 3)
        return ". ".join(sentences[:keep]) + "."
    elif method == 1:
        # Remove keywords: drop sentences containing common content words
        keywords = {"loved", "happy", "beautiful", "kind", "friend", "adventure",
                    "wonderful", "proud", "cheered", "smile", "brave", "magic"}
        filtered = [s for s in sentences
                    if not any(kw in s.lower() for kw in keywords)]
        if not filtered:
            filtered = sentences[:2]
        return ". ".join(filtered) + "."
    else:
        # Repetition: repeat the first sentence multiple times
        first = sentences[0] if sentences else "Once upon a time."
        repeats = rng.randint(2, 4)
        return ". ".join([first] * repeats) + "."


def _make_dpo_pairs(stories, num_samples, seed=42):
    """Generate DPO chosen/rejected pairs."""
    rng = random.Random(seed)
    pairs = []
    topics_pool = list(_TOPICS)

    for i in range(num_samples):
        story = stories[i % len(stories)]
        topic = topics_pool[i % len(topics_pool)]
        prompt = f"Write a short story about {topic}."
        chosen = story
        rejected = _corrupt_rejection(chosen, rng)
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return pairs[:num_samples]


# ---------------------------------------------------------------------------
# Eval prompt generators
# ---------------------------------------------------------------------------

_WORD_BANK = [
    ["cat", "ball", "garden"],
    ["dog", "park", "happy"],
    ["bird", "tree", "song"],
    ["fish", "river", "swim"],
    ["star", "night", "bright"],
    ["flower", "rain", "grow"],
    ["moon", "lake", "quiet"],
    ["bear", "honey", "forest"],
    ["rabbit", "carrot", "field"],
    ["sun", "beach", "sand"],
    ["cloud", "wind", "sky"],
    ["horse", "hill", "run"],
    ["frog", "pond", "jump"],
    ["snow", "mountain", "cold"],
    ["boat", "ocean", "wave"],
    ["butterfly", "spring", "colorful"],
    ["squirrel", "nut", "autumn"],
    ["rainbow", "rain", "beautiful"],
    ["elephant", "jungle", "big"],
    ["penguin", "ice", "dance"],
]

_STORY_TOPICS = [
    "a lost puppy", "a magical forest", "a day at the beach",
    "a friendly dragon", "a dancing robot", "a secret garden",
    "a flying car", "a talking tree", "a brave princess",
    "a silly monkey", "a rainbow bridge", "a tiny kingdom",
    "a singing mermaid", "a treasure map", "a space adventure",
    "a haunted castle", "a friendly ghost", "a circus elephant",
    "a flying kite", "a midnight picnic",
]

_OPENING_SENTENCES = [
    "Once upon a time there was a small village hidden between two mountains.",
    "The old clock in the tower struck midnight and something magical happened.",
    "A little girl named Emma found a golden key on her way to school.",
    "The animals in the forest were having their annual talent show.",
    "It was the rainiest day of the year and the river was rising fast.",
    "A mysterious package arrived at the door with no return address.",
    "The last leaf on the old oak tree finally let go of its branch.",
    "In the attic under a dusty blanket lay a book that glowed with light.",
    "The playground was empty except for one small child on the swings.",
    "A baker discovered that his bread could make people tell the truth.",
    "The lighthouse keeper saw something glowing at the bottom of the sea.",
    "Two friends found a map tucked inside an old library book.",
    "The school bus took a wrong turn and ended up in a fairy tale.",
    "A young inventor built a machine that could talk to animals.",
    "The first snow of winter fell on the little town by the lake.",
    "Grandma's recipe book contained one recipe written in disappearing ink.",
    "The garden gnome moved to a different spot every night when no one was looking.",
    "A boy discovered he could hear the thoughts of his pet hamster.",
    "The town fountain started bubbling with rainbow-colored water.",
    "An old pirate ship appeared in the harbor after a terrible storm.",
]


def _make_eval_prompts(num_total=20, seed=42):
    """Generate evaluation prompts of 5 types."""
    rng = random.Random(seed)
    prompts = []
    per_type = num_total // 5

    # 1. keyword_story
    for i in range(per_type):
        words = _WORD_BANK[i % len(_WORD_BANK)]
        prompts.append({
            "type": "keyword_story",
            "prompt": f"Write a short story using these words: {', '.join(words)}.",
            "required_words": words,
        })

    # 2. topic_story
    for i in range(per_type):
        topic = _STORY_TOPICS[i % len(_STORY_TOPICS)]
        emotion = rng.choice(_EMOTIONS[:3])  # happy, sad, funny
        prompts.append({
            "type": "topic_story",
            "prompt": f"Write a {emotion} story about {topic}.",
            "required_words": [],
        })

    # 3. continue_story
    for i in range(per_type):
        opening = _OPENING_SENTENCES[i % len(_OPENING_SENTENCES)]
        prompts.append({
            "type": "continue_story",
            "prompt": f"Continue the story: {opening}",
            "required_words": [],
        })

    # 4. style_control
    for i in range(per_type):
        topic = _STORY_TOPICS[(i + 5) % len(_STORY_TOPICS)]
        emotion = rng.choice(_EMOTIONS)
        prompts.append({
            "type": "style_control",
            "prompt": f"Write a {emotion} story about {topic}.",
            "required_words": [],
        })

    # 5. format_control
    for i in range(per_type):
        topic = _STORY_TOPICS[(i + 10) % len(_STORY_TOPICS)]
        prompts.append({
            "type": "format_control",
            "prompt": f"Write a story in exactly three sentences about {topic}.",
            "required_words": [],
        })

    return prompts[:num_total]


# ---------------------------------------------------------------------------
# File writing helpers
# ---------------------------------------------------------------------------

def _write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic toy data for MiniLLM pipeline."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Root output directory (default: data/)",
    )
    args = parser.parse_args()
    base = args.output_dir

    # ---- Raw text files ---------------------------------------------------
    raw_train = os.path.join(base, "raw", "toy_train.txt")
    raw_valid = os.path.join(base, "raw", "toy_valid.txt")
    _ensure_dir(raw_train)
    _ensure_dir(raw_valid)
    _write_lines(raw_train, TRAIN_STORIES)
    _write_lines(raw_valid, VALID_STORIES)
    print(f"[OK] {raw_train}  ({len(TRAIN_STORIES)} lines)")
    print(f"[OK] {raw_valid}  ({len(VALID_STORIES)} lines)")

    # ---- SFT JSONL --------------------------------------------------------
    sft_train = os.path.join(base, "sft", "train.jsonl")
    sft_valid = os.path.join(base, "sft", "valid.jsonl")
    _ensure_dir(sft_train)
    _ensure_dir(sft_valid)

    sft_train_pairs = _make_sft_pairs(TRAIN_STORIES, num_samples=50, seed=42)
    sft_valid_pairs = _make_sft_pairs(VALID_STORIES, num_samples=10, seed=123)
    _write_jsonl(sft_train, sft_train_pairs)
    _write_jsonl(sft_valid, sft_valid_pairs)
    print(f"[OK] {sft_train}  ({len(sft_train_pairs)} pairs)")
    print(f"[OK] {sft_valid}  ({len(sft_valid_pairs)} pairs)")

    # ---- DPO JSONL --------------------------------------------------------
    dpo_train = os.path.join(base, "dpo", "train.jsonl")
    dpo_valid = os.path.join(base, "dpo", "valid.jsonl")
    _ensure_dir(dpo_train)
    _ensure_dir(dpo_valid)

    dpo_train_pairs = _make_dpo_pairs(TRAIN_STORIES, num_samples=30, seed=42)
    dpo_valid_pairs = _make_dpo_pairs(VALID_STORIES, num_samples=10, seed=123)
    _write_jsonl(dpo_train, dpo_train_pairs)
    _write_jsonl(dpo_valid, dpo_valid_pairs)
    print(f"[OK] {dpo_train}  ({len(dpo_train_pairs)} pairs)")
    print(f"[OK] {dpo_valid}  ({len(dpo_valid_pairs)} pairs)")

    # ---- Eval prompts JSONL -----------------------------------------------
    eval_path = os.path.join(base, "prompts", "eval_prompts.jsonl")
    _ensure_dir(eval_path)

    eval_prompts = _make_eval_prompts(num_total=20, seed=42)
    _write_jsonl(eval_path, eval_prompts)
    print(f"[OK] {eval_path}  ({len(eval_prompts)} prompts)")

    print("\nDone. All toy data generated successfully.")


if __name__ == "__main__":
    main()
