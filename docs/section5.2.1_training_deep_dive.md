# Training Deep Dive: Understanding the Data Flow

This document explains how data flows through the World Model training pipeline, what each piece of data represents, and why it matters for predicting customer behavior.

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Understanding Temporal Data](#understanding-temporal-data)
3. [The Six Tensors Explained](#the-six-tensors-explained)
4. [Inside the Model: Encoder and Decoder](#inside-the-model-encoder-and-decoder)
5. [How Data Flows Through Training](#how-data-flows-through-training)
6. [The Training Process Step by Step](#the-training-process-step-by-step)
7. [Why Each Piece Matters](#why-each-piece-matters)
8. [Running the Training Pipeline](#running-the-training-pipeline)

> **Note**: This document covers **masked prediction** (BERT-style training). For **next-basket prediction** (recommended for RL/simulation), see [section5.2.2_next_basket_prediction.md](section5.2.2_next_basket_prediction.md).

---

## The Big Picture

### What Are We Trying to Do?

We're building a model that can answer: **"Given everything we know about a customer, what products will they buy in their next shopping trip?"**

To answer this, we need to understand:
- **Who** is shopping (customer history and preferences)
- **When** they're shopping (day, time, season)
- **Where** they're shopping (which store)
- **Why** they're shopping (quick trip vs. big weekly shop)
- **What** they typically buy (product patterns)

Each of these becomes a "tensor" - a structured package of numbers that the model can learn from.

### The Shopping Trip as Training Data

Every row in our training data represents a single shopping trip (basket). For each trip, we know:

```
Example Shopping Trip:
├── Customer: ID 12345 (has shopped 50 times before)
├── Date: Tuesday, March 15, 2007 at 10:30 AM
├── Store: Store #42 (large urban supermarket)
├── Trip Type: "Full Shop" (weekly grocery run)
├── Products Bought: [Milk, Bread, Eggs, Cheese, Apples, Chicken, Rice, ...]
└── Total Spent: £47.32
```

The model learns: "Customers like #12345, shopping on Tuesday mornings for a full shop, tend to buy these kinds of products."

---

## Understanding Temporal Data

### What is "Temporal" Data?

"Temporal" simply means "related to time." In our context, temporal data captures:

1. **When in the week**: Monday vs. Saturday shopping is different
2. **When in the day**: Morning shoppers vs. evening shoppers behave differently
3. **When in the year**: Christmas shopping vs. regular weeks
4. **Customer history timeline**: How long we've known this customer

### Why Does Time Matter So Much?

Shopping behavior follows strong temporal patterns:

```
Weekly Patterns:
┌─────────────────────────────────────────────────────────────┐
│  Mon   Tue   Wed   Thu   Fri   Sat   Sun                   │
│   ▁     ▂     ▂     ▃     ▅     █     ▆                    │
│                                                             │
│  Weekdays: Quick top-up shops (milk, bread)                │
│  Saturday: Big weekly shop (full trolley)                  │
│  Sunday: Relaxed browsing (treats, newspapers)             │
└─────────────────────────────────────────────────────────────┘

Daily Patterns:
┌─────────────────────────────────────────────────────────────┐
│  6am  9am  12pm  3pm  6pm  9pm                             │
│   ▁    ▃    ▅    ▃    █    ▂                               │
│                                                             │
│  Morning: Coffee, breakfast items, newspapers              │
│  Lunch: Ready meals, sandwiches                            │
│  Evening: Dinner ingredients (peak shopping)               │
└─────────────────────────────────────────────────────────────┘

Seasonal Patterns:
┌─────────────────────────────────────────────────────────────┐
│  Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec│
│   ▃    ▂    ▃    ▄    ▄    ▅    ▅    ▆    ▄    ▃    ▅    █ │
│                                                             │
│  Summer: BBQ items, salads, ice cream                      │
│  Winter: Soups, comfort food, hot drinks                   │
│  December: Huge spike for Christmas shopping               │
└─────────────────────────────────────────────────────────────┘
```

### The Temporal Split: Preventing "Time Travel"

We split our data chronologically to prevent the model from "cheating":

```
Timeline of Data:
════════════════════════════════════════════════════════════════

Week 1 ──────────────────────> Week 80 ───> Week 95 ───> Week 117
│                                    │           │            │
│◄────── TRAINING DATA ─────────────►│◄─ VALID ─►│◄── TEST ──►│
│                                    │           │            │
│  Learn patterns from               │  Tune     │  Final     │
│  historical shopping               │  model    │  evaluation│
│                                    │           │            │
════════════════════════════════════════════════════════════════

CRITICAL RULE: When predicting a basket at week W,
               the model can ONLY see data from weeks 1 to W-1.

               It cannot peek into the future!
```

**Why this matters**: In real life, we can't know what a customer will buy next week. The model must learn to predict using only past information, just like a real recommendation system would.

---

## The Six Tensors Explained

The model receives six packages of information (tensors) for each shopping trip. Think of these as six different "perspectives" on the shopping trip.

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE SIX TENSORS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  T1: CUSTOMER     "Who is this person?"                        │
│  ────────────                                                   │
│  192 numbers describing the customer                            │
│                                                                 │
│  T2: PRODUCTS     "What products are in this basket?"          │
│  ────────────                                                   │
│  256 numbers per product (their characteristics)                │
│                                                                 │
│  T3: TEMPORAL     "When is this happening?"                    │
│  ────────────                                                   │
│  64 numbers describing the time context                         │
│                                                                 │
│  T4: PRICE        "What do prices look like?"                  │
│  ────────────                                                   │
│  64 numbers per product (price patterns)                        │
│                                                                 │
│  T5: STORE        "Where is this happening?"                   │
│  ────────────                                                   │
│  96 numbers describing the store                                │
│                                                                 │
│  T6: TRIP         "What kind of shopping trip is this?"        │
│  ────────────                                                   │
│  48 numbers describing the mission                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### T1: Customer Context [192 dimensions]

**What it represents**: Everything we know about this specific customer.

```
T1 BREAKDOWN (192 numbers total):
══════════════════════════════════════════════════════════════

SEGMENT FEATURES [64 numbers]
────────────────────────────────
"What type of shopper is this?"

Examples encoded here:
• Demographic segment (family, single, elderly, etc.)
• Income bracket signals (premium vs. budget shopper)
• Lifestyle indicators (health-conscious, convenience-seeker)
• Shopping frequency pattern (weekly, sporadic, daily)

How it's used:
  A "family with kids" segment → higher probability for:
    - Cereal, snacks, juice boxes
    - Larger pack sizes
    - Kid-friendly products

HISTORY FEATURES [96 numbers]
────────────────────────────────
"What has this customer bought before?"

Encodes the customer's shopping history:
• Category preferences (loves dairy, rarely buys alcohol)
• Brand loyalty patterns (always buys Brand X)
• Basket size trends (spends ~£50 per trip)
• Seasonal behavior (stocks up before holidays)

How it's used:
  Customer who always buys milk → high probability for milk
  Customer who never buys fish → low probability for fish

AFFINITY FEATURES [32 numbers]
────────────────────────────────
"What products does this customer gravitate toward?"

Category affinity scores:
• Fresh produce lover vs. frozen food buyer
• Premium tier vs. economy tier preference
• Healthy choices vs. indulgent choices

How it's used:
  High "fresh produce" affinity → boost fresh items
  High "convenience" affinity → boost ready meals
```

**Real example**:

```
Customer #12345's T1 encoding might represent:
┌──────────────────────────────────────────────────────────┐
│ "Middle-income family shopper, visits weekly, prefers   │
│  branded products, high affinity for dairy and bakery,  │
│  typically spends £60-80, usually shops with kids,      │
│  responds to promotions, loyal to this store chain"     │
└──────────────────────────────────────────────────────────┘
```

---

### T2: Product Embeddings [256 dimensions per product]

**What it represents**: The "DNA" of each product - everything that makes it unique.

```
T2: PRODUCT EMBEDDING STRUCTURE
══════════════════════════════════════════════════════════════

Each product gets 256 numbers that capture:

CATEGORY INFORMATION [~64 dimensions]
────────────────────────────────
Where does this product live in the store?

Hierarchy encoded:
  Department → Aisle → Shelf → Category → Subcategory

Example: "Semi-Skimmed Milk 2L"
  • Department: Dairy
  • Aisle: Chilled
  • Category: Milk
  • Subcategory: Fresh Milk
  • Pack Type: Large format

CO-PURCHASE PATTERNS [~64 dimensions]
────────────────────────────────
What products are bought together?

Learned from shopping baskets:
  • Bread ↔ Butter (often together)
  • Pasta ↔ Pasta Sauce (frequently paired)
  • Wine ↔ Cheese (complementary)

This creates "neighborhoods" of related products

SUBSTITUTION PATTERNS [~64 dimensions]
────────────────────────────────
What products replace each other?

When one is unavailable, customers buy:
  • Coca-Cola ↔ Pepsi (direct substitutes)
  • Butter ↔ Margarine (alternatives)
  • Beef ↔ Chicken (protein options)

BRAND/TIER INFORMATION [~64 dimensions]
────────────────────────────────
Premium, mid-range, or economy?

Encoded brand positioning:
  • Premium: Waitrose Duchy, Finest ranges
  • Mid-tier: Standard branded products
  • Economy: Value ranges, own-brand basics
```

**How products are related**:

```
PRODUCT EMBEDDING SPACE (simplified 2D view):

           Premium ↑
                   │    ▲ Organic Milk
                   │         ▲ Branded Milk
                   │              ▲ Standard Milk
                   │                   ▲ Value Milk
   ────────────────┼────────────────────────────────→
   Healthy         │                        Indulgent
                   │
     ▲ Salad       │              ▲ Crisps
        ▲ Fruit    │         ▲ Chocolate
                   │    ▲ Biscuits
                   │
           Economy ↓

Products near each other in this space:
• Are often bought together
• Can substitute for each other
• Appeal to similar customers
```

---

### T3: Temporal Context [64 dimensions]

**What it represents**: Everything about WHEN this shopping trip is happening.

```
T3 BREAKDOWN (64 numbers total):
══════════════════════════════════════════════════════════════

WEEK OF YEAR [16 numbers]
────────────────────────────────
Which week is it? (1-52)

Encoded using sine/cosine waves to capture:
• Seasonality (week 25 = summer, week 50 = Christmas)
• Cyclical nature (week 52 is close to week 1)

Effect on predictions:
  Week 51-52: ↑ Turkey, stuffing, mince pies
  Week 25-30: ↑ BBQ items, salads, ice cream

DAY OF WEEK [8 numbers]
────────────────────────────────
Monday (1) through Sunday (7)

Captures weekly shopping rhythm:
  Monday:    Recovery from weekend, basic needs
  Tuesday:   Normal shopping
  Wednesday: Mid-week top-up
  Thursday:  Pre-weekend preparation starts
  Friday:    Weekend preparation, treats
  Saturday:  Big weekly shop
  Sunday:    Relaxed browsing, newspapers

Effect on predictions:
  Saturday: ↑ Full basket predictions
  Monday:   ↑ Quick essentials predictions

HOUR OF DAY [8 numbers]
────────────────────────────────
0-23 hour encoding

Captures daily patterns:
  7-9 AM:   Breakfast items, coffee, newspapers
  12-2 PM:  Lunch items, sandwiches
  5-7 PM:   Dinner ingredients (PEAK)
  8-10 PM:  Last-minute needs, ready meals

Effect on predictions:
  7 AM:   ↑ Coffee, pastries, newspapers
  6 PM:   ↑ Fresh meat, vegetables

HOLIDAY INDICATOR [8 numbers]
────────────────────────────────
Is this a special period?

Flags for:
  • Christmas season (weeks 50-52)
  • Easter period
  • Bank holidays
  • School holidays (summer, half-terms)

Effect on predictions:
  Christmas: ↑ Alcohol, chocolate, party food
  Easter:    ↑ Eggs (chocolate & real), lamb

SEASON [8 numbers]
────────────────────────────────
Spring, Summer, Autumn, Winter

Broad seasonal patterns:
  Spring: Fresh starts, lighter foods
  Summer: Outdoor eating, cold drinks
  Autumn: Comfort food begins
  Winter: Hearty meals, warming foods

TREND & RECENCY [16 numbers]
────────────────────────────────
Position in the dataset timeline

Captures:
• How far into the year are we?
• Long-term trends in shopping
• Recency of customer's last visit
```

**Why temporal encoding uses sine/cosine waves**:

```
THE CYCLICAL TIME PROBLEM:
══════════════════════════════════════════════════════════════

Naive encoding:  Week 52 = 52,  Week 1 = 1
Problem:         Model thinks week 52 and week 1 are far apart!
Reality:         They're actually neighbors (Dec 31 → Jan 1)

Solution: Sine/Cosine encoding creates circular representation

Week numbers as a circle:
                 Week 1
                   │
          Week 52  │  Week 2
               \   │   /
                \  │  /
    Week 40 ─────( ● )───── Week 13
                /  │  \
               /   │   \
          Week 30  │  Week 20
                   │
                Week 26

Now: Week 52 and Week 1 are NEIGHBORS (correct!)
     Week 26 is OPPOSITE to Week 1 (correct!)
```

---

### T4: Price Features [64 dimensions per product]

**What it represents**: Price signals that influence buying decisions.

```
T4 BREAKDOWN (64 numbers per product):
══════════════════════════════════════════════════════════════

CURRENT PRICE POSITION [16 numbers]
────────────────────────────────
Where is the price right now vs. history?

Encodes:
• Is this cheap/normal/expensive for this product?
• Percentile ranking (top 10% cheapest ever?)
• Price relative to similar products

Effect: Low price position → ↑ purchase probability

PRICE VOLATILITY [16 numbers]
────────────────────────────────
How much does this price jump around?

Encodes:
• Stable prices (basic commodities like milk)
• Volatile prices (fresh produce, seasonal items)
• Promotional frequency

Effect: High volatility + current low → BUY NOW signal

PROMOTIONAL SIGNALS [16 numbers]
────────────────────────────────
Is there a deal happening?

Detects:
• Multi-buy offers (3 for 2)
• Percentage discounts
• Price-drop patterns
• Clearance signals

Effect: Active promotion → ↑ purchase probability

CROSS-CATEGORY PRICE CONTEXT [16 numbers]
────────────────────────────────
How do prices compare across the basket?

Captures:
• Is the whole basket expensive/cheap today?
• Budget remaining signals
• Category-level price trends

Effect: If meat is expensive → ↑ vegetarian alternatives
```

**How price features are generated (Fourier encoding)**:

```
RAW PRICE HISTORY:
══════════════════════════════════════════════════════════════

Product: "Branded Cola 2L"

Week:  1   2   3   4   5   6   7   8   9  10  11  12
Price: £2  £2  £2  £1.50  £1.50  £2  £2  £2  £2  £1  £1  £2
           └─────────┘              └─────────┘
           Promotion                 Deep discount

Fourier transformation extracts patterns:
┌─────────────────────────────────────────────────────────────┐
│ • Base price level: ~£1.85                                  │
│ • Promotion frequency: Every 3-4 weeks                      │
│ • Promotion depth: ~25% off                                 │
│ • Current position: Week 10 = deep discount = BUY SIGNAL    │
└─────────────────────────────────────────────────────────────┘

These patterns become the 64 price feature numbers.
```

---

### T5: Store Context [96 dimensions]

**What it represents**: Everything about WHERE this shopping is happening.

```
T5 BREAKDOWN (96 numbers total):
══════════════════════════════════════════════════════════════

STORE TYPE [24 numbers]
────────────────────────────────
What kind of store is this?

Categories:
• Hypermarket: Huge selection, full range
• Supermarket: Standard full-service
• Metro/Express: Small, convenience focused
• Online: Different browsing behavior

Effect: Metro store → ↓ bulk items, ↑ quick meals

LOCATION DEMOGRAPHICS [24 numbers]
────────────────────────────────
Who lives near this store?

Encodes local population:
• Urban vs. suburban vs. rural
• Income levels of catchment area
• Family composition (students, families, retirees)
• Ethnic composition (affects product range)

Effect: Student area → ↑ cheap meals, ↓ family packs

PRODUCT ASSORTMENT [24 numbers]
────────────────────────────────
What does this store actually stock?

Captures:
• Range depth (how many variants?)
• Category strengths (great deli, limited frozen)
• Local specialties
• Missing categories

Effect: No fresh fish counter → ↓ fresh fish predictions

STORE PERFORMANCE [24 numbers]
────────────────────────────────
How does this store perform?

Includes:
• Sales velocity (busy vs. quiet store)
• Basket size norms for this store
• Popular categories here
• Promotional responsiveness

Effect: Quiet store → different shopping patterns
```

**Example store profiles**:

```
Store Type Comparison:
══════════════════════════════════════════════════════════════

HYPERMARKET (out of town)          METRO (city center)
─────────────────────────          ────────────────────
• Full range: 40,000 products      • Limited: 3,000 products
• Average basket: £80              • Average basket: £12
• Trip type: Weekly shop           • Trip type: Daily top-up
• Customer: Families with cars     • Customer: Office workers
• Key categories:                  • Key categories:
  - Bulk buys                        - Sandwiches/ready meals
  - Household goods                  - Drinks/snacks
  - Full fresh range                 - Essentials only

The model learns: Same customer behaves DIFFERENTLY in different stores!
```

---

### T6: Trip Context [48 dimensions]

**What it represents**: What KIND of shopping trip this is.

```
T6 BREAKDOWN (48 numbers total):
══════════════════════════════════════════════════════════════

MISSION TYPE [16 numbers]
────────────────────────────────
Why is the customer here?

Categories:
• Full Shop:    Weekly stock-up, full trolley
• Top-Up:       Quick trip for a few items
• Small Shop:   Mid-week smaller shop
• Emergency:    Forgot something, urgent need

Effect on predictions:
  Full Shop → Expect 30-50 products across all categories
  Top-Up    → Expect 3-5 products, essentials only

MISSION FOCUS [16 numbers]
────────────────────────────────
What's driving this trip?

Categories:
• Fresh:    Fruits, veg, meat, bakery
• Grocery:  Cupboard staples
• Mixed:    Combination
• Nonfood:  Household, toiletries

Effect on predictions:
  Fresh focus → ↑ produce, meat, dairy predictions
  Grocery focus → ↑ canned, dried, packaged goods

PRICE SENSITIVITY [8 numbers]
────────────────────────────────
How price-conscious is this trip?

Levels:
• LA (Less Affluent): Very price sensitive
• MM (Mainstream):    Balanced considerations
• UM (Upmarket):      Quality over price

Effect on predictions:
  LA → ↑ value products, own-brand, promotions
  UM → ↑ premium products, branded, organic

BASKET SIZE INDICATOR [8 numbers]
────────────────────────────────
How big will this basket be?

Categories:
• S (Small):  1-5 items
• M (Medium): 6-15 items
• L (Large):  16+ items

Effect on predictions:
  Large basket → Predict more variety
  Small basket → Focus on essentials
```

**Mission type examples**:

```
THE SAME CUSTOMER, DIFFERENT MISSIONS:
══════════════════════════════════════════════════════════════

SATURDAY 10 AM: Full Shop
─────────────────────────
Mission: Weekly grocery run
Basket:  40 items, £85
Products: Full range - fresh, frozen, household
          Milk, bread, veg, meat, cleaning supplies...

TUESDAY 6 PM: Top-Up
─────────────────────────
Mission: Ran out of milk
Basket:  3 items, £8
Products: Just essentials
          Milk, bread, maybe a snack

FRIDAY 5 PM: Special Occasion
─────────────────────────
Mission: Dinner party prep
Basket:  15 items, £60
Products: Premium focus
          Wine, cheese, steak, fancy dessert

The T6 tensor helps the model understand WHAT to expect!
```

---

## Inside the Model: Encoder and Decoder

Now that we understand the six input tensors, let's look at what happens INSIDE the model. The World Model has two main parts:

1. **Mamba Encoder**: Processes the customer's shopping context
2. **Transformer Decoder**: Generates product predictions

Think of it like this:
- The **Encoder** reads and understands the situation ("who, when, where, why")
- The **Decoder** writes the answer ("what products will they buy")

### The Model at a Glance

```
THE WORLD MODEL ARCHITECTURE
══════════════════════════════════════════════════════════════════════════

                    INPUTS                          OUTPUTS
                    ──────                          ───────

    ┌─────────┐
    │ T1:Cust │──┐
    │  [192]  │  │
    └─────────┘  │
    ┌─────────┐  │    ┌──────────────────┐
    │ T3:Time │──┼───▶│  CONTEXT FUSION  │
    │  [64]   │  │    │  Combine into    │
    └─────────┘  │    │  single vector   │
    ┌─────────┐  │    │                  │
    │ T5:Store│──┤    │  [400] → [512]   │
    │  [96]   │  │    └────────┬─────────┘
    └─────────┘  │             │
    ┌─────────┐  │             │
    │ T6:Trip │──┘             │         ┌─────────────────────────────┐
    │  [48]   │                │         │                             │
    └─────────┘                │         │      MAMBA ENCODER          │
                               │         │      ───────────────        │
    ┌─────────┐                │         │                             │
    │T2:Prods │──┐             ▼         │  "Understand the customer   │
    │[S×256]  │  │    ┌──────────────┐   │   and their situation"      │
    └─────────┘  ├───▶│PRODUCT FUSION│──▶│                             │
    ┌─────────┐  │    │ [S×320]→[S×512]  │  4 layers of processing     │
    │T4:Price │──┘    └──────────────┘   │  O(n) linear complexity     │
    │[S×64]   │                          │                             │
    └─────────┘                          │  OUTPUT: Customer State     │
                                         │          [S+1 × 512]        │
                                         └──────────────┬──────────────┘
                                                        │
                                                        │ "Here's what I
                                                        │  understand about
                                                        │  this customer"
                                                        ▼
                                         ┌─────────────────────────────┐
                                         │                             │
                                         │   TRANSFORMER DECODER       │
                                         │   ────────────────────      │
                                         │                             │
                                         │  "Predict what products     │
                                         │   belong in this basket"    │
                                         │                             │
                                         │  2 layers with:             │
                                         │  • Self-attention           │
                                         │  • Cross-attention to       │
                                         │    encoder output           │
                                         │                             │
                                         │  OUTPUT: Product Logits     │
                                         │          [B × 5003]         │
                                         └──────────────┬──────────────┘
                                                        │
                                                        ▼
                                         ┌─────────────────────────────┐
                                         │       OUTPUT HEADS          │
                                         │                             │
                                         │  • Product Prediction       │
                                         │    "P(milk) = 0.73"         │
                                         │                             │
                                         │  • Basket Size              │
                                         │    "Large shop"             │
                                         │                             │
                                         │  • Price Sensitivity        │
                                         │    "Budget conscious"       │
                                         │                             │
                                         │  • Mission Type             │
                                         │    "Weekly stock-up"        │
                                         └─────────────────────────────┘
```

---

### Step 1: Context Fusion (Preparing the Inputs)

Before anything goes into the encoder, we need to combine our tensors into a format the model can work with.

```
CONTEXT FUSION: Combining Dense Tensors
══════════════════════════════════════════════════════════════════════════

STEP 1: Concatenate the "who/when/where/why" tensors
────────────────────────────────────────────────────

    T1: Customer [192]  ─┐
    T3: Temporal [64]   ─┼──▶ Concatenate ──▶ [400 numbers]
    T5: Store    [96]   ─┤
    T6: Trip     [48]   ─┘

    This 400-number vector captures:
    "A dairy-loving family shopper, visiting on Saturday morning,
     at a large suburban store, doing a full weekly shop"


STEP 2: Project to model dimension
────────────────────────────────────────────────────

    [400] ──▶ Linear Layer ──▶ LayerNorm ──▶ GELU ──▶ [512]

    Why 512? This is the "language" the model thinks in.
    All internal representations are 512-dimensional.


PRODUCT FUSION: Combining Sequence Tensors
══════════════════════════════════════════════════════════════════════════

For each product in the basket:

    T2: Product Embedding [256]  ─┐
                                  ├──▶ Concatenate ──▶ [320] ──▶ Project ──▶ [512]
    T4: Price Features    [64]   ─┘

    Plus: Add positional encoding (so model knows product ORDER)

    Result: Each product becomes a 512-dimensional vector

Example basket with 6 products:
    Position 0: [Context vector - 512 numbers]  ← "CLS token" (represents whole basket)
    Position 1: [Milk - 512 numbers]
    Position 2: [Bread - 512 numbers]
    Position 3: [Eggs - 512 numbers]
    Position 4: [Cheese - 512 numbers]
    Position 5: [Apples - 512 numbers]
    Position 6: [Chicken - 512 numbers]

    Shape: [7 positions × 512 dimensions]
```

**What goes IN to fusion:**
- T1 (192), T3 (64), T5 (96), T6 (48) = 400 dense numbers
- T2 (S × 256), T4 (S × 64) = S × 320 sequence numbers

**What comes OUT of fusion:**
- Context vector: [512] numbers
- Product sequence: [S × 512] numbers
- Combined: [(S+1) × 512] - the context vector becomes position 0

---

### Step 2: The Mamba Encoder

The Mamba Encoder is the "brain" that understands the customer's situation. It uses a special architecture called a "State Space Model" that's incredibly efficient at processing sequences.

```
MAMBA ENCODER: Understanding the Customer
══════════════════════════════════════════════════════════════════════════

INPUT:  Fused sequence [(S+1) × 512]
        Position 0: Context (who/when/where/why)
        Positions 1-S: Products in basket

OUTPUT: Encoded sequence [(S+1) × 512]
        Same shape, but now each position "understands" everything else


WHY MAMBA? The Efficiency Story
────────────────────────────────────────────────────

Traditional Transformers: O(n²) complexity
  For 100 items: 100 × 100 = 10,000 operations
  For 500 items: 500 × 500 = 250,000 operations  ← Gets expensive fast!

Mamba (State Space Model): O(n) complexity
  For 100 items: 100 operations
  For 500 items: 500 operations  ← Scales gracefully!

This matters because:
  • Customer histories can span 100+ weeks
  • Each week might have multiple shopping trips
  • We need to process millions of baskets during training


HOW MAMBA WORKS (Simplified)
────────────────────────────────────────────────────

Think of Mamba as reading a book with a "memory" that updates as it reads:

    Reading position 1 (Milk):
    ┌─────────────────────────────────────────────────────────────┐
    │ Memory = "I've seen milk"                                   │
    │ Output = "Milk, knowing nothing else yet"                   │
    └─────────────────────────────────────────────────────────────┘

    Reading position 2 (Bread):
    ┌─────────────────────────────────────────────────────────────┐
    │ Memory = "I've seen milk, now bread"                        │
    │ Output = "Bread, knowing it follows milk"                   │
    │          → "These often go together! Breakfast basket?"     │
    └─────────────────────────────────────────────────────────────┘

    Reading position 3 (Eggs):
    ┌─────────────────────────────────────────────────────────────┐
    │ Memory = "Milk, bread, eggs..."                             │
    │ Output = "Eggs, knowing the full context"                   │
    │          → "Definitely a breakfast/baking basket!"          │
    └─────────────────────────────────────────────────────────────┘

The "Selective" Part:
  Mamba can choose what to remember and what to forget.

  Important event (big shopping trip): "Remember this clearly!"
  Routine event (bought milk again): "Just update the pattern slightly"

  This is controlled by the "dt" (delta time) parameter:
  • Small dt = "Focus on this, it's important"
  • Large dt = "Fast-forward, nothing new here"


THE FOUR MAMBA LAYERS
────────────────────────────────────────────────────

Layer 1: Basic patterns
  "Milk and bread go together"
  "This is a morning shop"

Layer 2: Category understanding
  "This is a fresh-focused basket"
  "Customer prefers branded products"

Layer 3: Behavioral patterns
  "Customer responds to promotions"
  "Typical basket size is medium"

Layer 4: Holistic understanding
  "This is a loyal customer doing their usual Saturday shop,
   focused on fresh items, somewhat price-sensitive"


MAMBA OUTPUT
────────────────────────────────────────────────────

Input:  Raw features that don't "know" about each other
Output: Contextual representations where every position
        understands the full picture

    Before Mamba:
        Milk [512] = "I am milk, a dairy product"
        Bread [512] = "I am bread, a bakery product"

    After Mamba:
        Milk [512] = "I am milk, in a breakfast basket with bread and eggs,
                      bought by a family shopper on Saturday morning,
                      at a store that usually has good dairy prices"
        Bread [512] = "I am bread, complementing milk and eggs,
                      in what looks like a full weekly shop..."
```

**What goes IN to Mamba Encoder:**
- Shape: `[Batch × (S+1) × 512]`
- Content: Context vector + product embeddings (no cross-knowledge yet)

**What comes OUT of Mamba Encoder:**
- Shape: `[Batch × (S+1) × 512]` (same shape!)
- Content: Same positions, but now each one "understands" the whole sequence
- The output at position 0 (context) now encodes the ENTIRE customer state

---

### Step 3: The Transformer Decoder

While Mamba is great at efficiently processing long sequences, the Transformer Decoder excels at one specific thing: **asking questions and getting answers**.

```
TRANSFORMER DECODER: Predicting Products
══════════════════════════════════════════════════════════════════════════

INPUT:
  • Query: "What product should go in position X?"
  • Memory: The Mamba encoder's output (customer understanding)

OUTPUT:
  • Answer: Probability distribution over all 5003 products


WHY TRANSFORMER FOR DECODING?
────────────────────────────────────────────────────

The key feature is CROSS-ATTENTION: the ability to "ask questions"
about the customer state and get relevant answers.

Example: Predicting position 4 (after Milk, Bread, Eggs)

    Query from Decoder:
    "I need to predict position 4. I already see Milk, Bread, Eggs.
     What would fit here?"

    Cross-Attention to Encoder Output:
    ┌─────────────────────────────────────────────────────────────┐
    │ Decoder asks: "What does this customer usually buy         │
    │                with milk, bread, and eggs?"                 │
    │                                                             │
    │ Encoder responds with relevant memories:                    │
    │   • Customer context: "Family shopper, full shop mission"   │
    │   • Pattern: "This customer often buys butter with these"   │
    │   • Store: "This store has butter on promotion today"       │
    │                                                             │
    │ Decoder concludes: "Butter is very likely!" (P=0.65)        │
    └─────────────────────────────────────────────────────────────┘

This "question and answer" mechanism is called ATTENTION.


THE TWO TYPES OF ATTENTION IN THE DECODER
────────────────────────────────────────────────────

1. SELF-ATTENTION (Causal/Masked)
   "Look at what I've already predicted"

   When predicting position 4:
     Can see:    Position 0 (context), 1 (milk), 2 (bread), 3 (eggs)
     Cannot see: Position 4, 5, 6, ... (haven't predicted yet!)

   Causal Mask:
        Pos:  0   1   2   3   4   5
          0 [ ✓   ✗   ✗   ✗   ✗   ✗ ]
          1 [ ✓   ✓   ✗   ✗   ✗   ✗ ]
          2 [ ✓   ✓   ✓   ✗   ✗   ✗ ]
          3 [ ✓   ✓   ✓   ✓   ✗   ✗ ]
          4 [ ✓   ✓   ✓   ✓   ✓   ✗ ]  ← Position 4 sees 0-4
          5 [ ✓   ✓   ✓   ✓   ✓   ✓ ]

   ✓ = Can attend (see this position)
   ✗ = Cannot attend (masked out)


2. CROSS-ATTENTION
   "Ask the encoder about the customer"

   The decoder sends a QUERY: "What should go here?"
   The encoder provides KEYS and VALUES: "Here's what I know"

   Cross-Attention Visualization:
   ┌────────────────────────────────────────────────────────────────┐
   │                                                                │
   │    DECODER                         ENCODER                     │
   │    (What to predict)               (What we know)              │
   │                                                                │
   │    Position 4: [Query]  ─────────▶ Position 0: [Customer]     │
   │         │              ╲           Position 1: [Milk]         │
   │         │               ╲          Position 2: [Bread]        │
   │         │                ─────────▶Position 3: [Eggs]         │
   │         │                                                      │
   │         │     Attention Weights:                               │
   │         │     Customer: 0.35 ← "Trip context matters most"    │
   │         │     Milk: 0.25      ← "What goes with milk?"        │
   │         │     Bread: 0.20     ← "Breakfast pattern"           │
   │         │     Eggs: 0.20      ← "Completes the pattern"       │
   │         │                                                      │
   │         ▼                                                      │
   │    [Weighted combination of encoder knowledge]                 │
   │         │                                                      │
   │         ▼                                                      │
   │    "Butter is likely" (high score)                            │
   │    "Cheese is possible" (medium score)                        │
   │    "Fish is unlikely" (low score)                             │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘


A SINGLE DECODER LAYER
────────────────────────────────────────────────────

    Input: [S+1 × 512]
           │
           ▼
    ┌──────────────────────────┐
    │    SELF-ATTENTION        │ "What have I predicted so far?"
    │    (with causal mask)    │
    └──────────┬───────────────┘
               │ + Residual + LayerNorm
               ▼
    ┌──────────────────────────┐
    │    CROSS-ATTENTION       │ "What does the encoder know?"
    │    (to encoder output)   │ ←──── Encoder Output [S+1 × 512]
    └──────────┬───────────────┘
               │ + Residual + LayerNorm
               ▼
    ┌──────────────────────────┐
    │    FEED-FORWARD          │ "Process this information"
    │    (512 → 2048 → 512)    │
    └──────────┬───────────────┘
               │ + Residual + LayerNorm
               ▼
    Output: [S+1 × 512]

    This is repeated 2 times (2 decoder layers).


DECODER OUTPUT
────────────────────────────────────────────────────

After 2 layers of processing, each position has a 512-dimensional
representation that captures:
  • What products came before (self-attention)
  • Customer preferences and context (cross-attention)
  • Complex patterns and interactions (feed-forward)

Shape: [Batch × (S+1) × 512]
```

**What goes IN to Transformer Decoder:**
- Encoder output: `[Batch × (S+1) × 512]` (the customer understanding)
- Decoder input: Same as encoder input (for self-decoding during training)

**What comes OUT of Transformer Decoder:**
- Shape: `[Batch × (S+1) × 512]`
- Content: Representations ready for prediction at each position

---

### Step 4: Output Heads (Making Predictions)

The final step converts the decoder's 512-dimensional representations into actual predictions.

```
OUTPUT HEADS: From Representations to Predictions
══════════════════════════════════════════════════════════════════════════

The decoder output [S+1 × 512] goes through specialized "heads"
that make different types of predictions.


HEAD 1: PRODUCT PREDICTION (Main Task)
────────────────────────────────────────────────────

For each MASKED position, predict which product belongs there.

    Decoder output at masked position: [512]
           │
           ▼
    ┌──────────────────────────┐
    │  Linear: 512 → 512       │
    │  GELU activation         │
    │  LayerNorm               │
    │  Linear: 512 → 5003      │  ← 5003 = all products + special tokens
    └──────────┬───────────────┘
               │
               ▼
    Logits: [5003]

    After Softmax:
    ┌─────────────────────────────────────────────────────────────┐
    │  Product 1 (Milk):     0.73   ← Most likely!                │
    │  Product 2 (Bread):    0.12                                 │
    │  Product 3 (Butter):   0.08                                 │
    │  Product 4 (Eggs):     0.03                                 │
    │  ...                                                        │
    │  Product 5000:         0.00001                              │
    │  [PAD]:                0.0                                  │
    │  [EOS]:                0.0                                  │
    │  [MASK]:               0.0                                  │
    └─────────────────────────────────────────────────────────────┘

    Prediction: "This masked position should be MILK"


HEAD 2-5: AUXILIARY PREDICTIONS (Helper Tasks)
────────────────────────────────────────────────────

These use the FIRST position (position 0, the "CLS" token)
which represents the entire basket.

    Decoder output at position 0: [512]
           │
           ├──▶ Basket Size Head:      [512] → [4]  (S/M/L/XL)
           │
           ├──▶ Price Sensitivity Head: [512] → [4]  (LA/MM/UM/Premium)
           │
           ├──▶ Mission Type Head:      [512] → [5]  (Full/Top-up/Small/Emergency/Other)
           │
           └──▶ Mission Focus Head:     [512] → [6]  (Fresh/Grocery/Mixed/Nonfood/General/Other)


    Example predictions for a basket:
    ┌─────────────────────────────────────────────────────────────┐
    │  Basket Size:        Large (0.78)                          │
    │  Price Sensitivity:  Mainstream (0.65)                      │
    │  Mission Type:       Full Shop (0.82)                       │
    │  Mission Focus:      Mixed (0.54)                           │
    └─────────────────────────────────────────────────────────────┘

    These auxiliary tasks help the model learn better representations
    by forcing it to understand the overall context, not just products.
```

**What goes IN to Output Heads:**
- Decoder output: `[Batch × (S+1) × 512]`
- Masked positions (which positions to predict)

**What comes OUT of Output Heads:**
- Product logits: `[Batch × num_masked × 5003]` - probability over all products
- Basket size: `[Batch × 4]`
- Price sensitivity: `[Batch × 4]`
- Mission type: `[Batch × 5]`
- Mission focus: `[Batch × 6]`

---

### Summary: The Full Journey

```
COMPLETE INPUT → OUTPUT JOURNEY
══════════════════════════════════════════════════════════════════════════

1. RAW INPUTS
   ────────────────────────────
   T1: Customer [192]     "Who"
   T2: Products [S × 256] "What products"
   T3: Temporal [64]      "When"
   T4: Prices [S × 64]    "Price context"
   T5: Store [96]         "Where"
   T6: Trip [48]          "Why/Mission"

2. CONTEXT FUSION
   ────────────────────────────
   Dense: T1+T3+T5+T6 [400] → [512]    "The shopping context"
   Sequence: T2+T4 [S×320] → [S×512]   "The products with prices"
   Combined: [(S+1) × 512]              "Context + products"

3. MAMBA ENCODER
   ────────────────────────────
   Input:  [(S+1) × 512]  "Raw features"
   Output: [(S+1) × 512]  "Contextual understanding"

   Each position now "knows" about all other positions.
   The model understands the customer's situation.

4. TRANSFORMER DECODER
   ────────────────────────────
   Input:  [(S+1) × 512]  "What to decode"
   Memory: [(S+1) × 512]  "Encoder's knowledge"
   Output: [(S+1) × 512]  "Ready for prediction"

   Cross-attention allows "asking questions" to the encoder.
   Causal mask ensures we don't peek at future positions.

5. OUTPUT HEADS
   ────────────────────────────
   Input:  [(S+1) × 512]  "Decoder output"

   Output:
   • Masked product predictions: [num_masked × 5003]
     "Position 3 should be BUTTER (73% confident)"

   • Auxiliary predictions (from position 0):
     "This is a LARGE basket (78%)"
     "MAINSTREAM price sensitivity (65%)"
     "FULL SHOP mission (82%)"

6. LOSS CALCULATION
   ────────────────────────────
   Compare predictions to reality:
   • Focal Loss: Were product predictions correct?
   • Contrastive Loss: Do similar products have similar embeddings?
   • Auxiliary Losses: Were basket/mission predictions correct?

   Total Loss → Backpropagation → Model learns!
```

---

## How Data Flows Through Training

### The Complete Picture

```
DATA FLOW: From Raw Transaction to Prediction
══════════════════════════════════════════════════════════════════════════

RAW DATA (what we start with)
─────────────────────────────
transactions.csv:
│ BASKET_ID │ CUST_CODE │ STORE_CODE │ PROD_CODE │ SHOP_WEEK │ PRICE │...│
│  1000001  │   C_12345 │    S_042   │   P_1234  │   200612  │  2.49 │...│
│  1000001  │   C_12345 │    S_042   │   P_5678  │   200612  │  1.99 │...│
│  1000001  │   C_12345 │    S_042   │   P_9999  │   200612  │  3.49 │...│

                            │
                            ▼
                     ┌──────────────┐
                     │   STAGE 1:   │
                     │ Feature Eng. │
                     └──────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌─────────┐           ┌─────────┐           ┌─────────┐
│Customer │           │Product  │           │ Price   │
│Histories│           │ Graph   │           │Features │
│  .npy   │           │  .npy   │           │.parquet │
└─────────┘           └─────────┘           └─────────┘
    │                       │                       │
    │                       │                       │
    └───────────────────────┼───────────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   STAGE 2:   │
                     │ Tensor Cache │
                     └──────────────┘
                            │
    ┌───────────┬───────────┼───────────┬───────────┐
    │           │           │           │           │
    ▼           ▼           ▼           ▼           ▼
┌───────┐  ┌───────┐   ┌───────┐  ┌───────┐   ┌───────┐
│  T1   │  │  T2   │   │  T3   │  │  T5   │   │  T6   │
│ Cust  │  │ Prod  │   │ Time  │  │ Store │   │ Trip  │
└───────┘  └───────┘   └───────┘  └───────┘   └───────┘
    │           │           │           │           │
    └───────────┴───────────┼───────────┴───────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   DATASET    │
                     │   LOADER     │
                     └──────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │        BATCH CREATION       │
              │                             │
              │  For 256 shopping trips:    │
              │  • T1: [256, 192]           │
              │  • T2: [256, 50, 256]       │
              │  • T3: [256, 64]            │
              │  • T4: [256, 50, 64]        │
              │  • T5: [256, 96]            │
              │  • T6: [256, 48]            │
              └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │      WORLD MODEL            │
              │                             │
              │  1. Fuse dense context      │
              │     T1+T3+T5+T6 → [512]     │
              │                             │
              │  2. Fuse products           │
              │     T2+T4 → [512 per prod]  │
              │                             │
              │  3. Mamba Encoder           │
              │     Learn customer state    │
              │                             │
              │  4. Transformer Decoder     │
              │     Predict products        │
              └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │       PREDICTIONS           │
              │                             │
              │  For each masked position:  │
              │  "This should be MILK"      │
              │  P(milk) = 0.73             │
              │  P(bread) = 0.12            │
              │  P(eggs) = 0.08             │
              │  ...                        │
              └─────────────────────────────┘
```

---

### Batch Creation: What Actually Gets Fed to the Model

```
INSIDE A TRAINING BATCH
══════════════════════════════════════════════════════════════════════════

A batch contains 256 shopping trips (baskets).
Here's what one sample looks like:

SAMPLE #47 in batch:
─────────────────────
Customer: C_12345
Store: S_042
Date: Tuesday, March 15, 2007, 10:30 AM
Week: 80 (training set)

Actual basket: [Milk, Bread, Eggs, Cheese, Apples, Chicken]

TENSOR VALUES FOR THIS SAMPLE:
────────────────────────────────────────────────────────────────

T1: Customer Context [192 numbers]
    [0.23, -0.15, 0.87, 0.03, ...] ← Encodes: "Family shopper,
                                              weekly visits,
                                              dairy lover"

T2: Product Embeddings [6 products × 256 numbers]
    Milk:    [0.45, 0.12, -0.33, ...]  ← "Dairy, staple, branded"
    Bread:   [0.38, 0.21, -0.19, ...]  ← "Bakery, staple, daily"
    Eggs:    [0.42, 0.08, -0.28, ...]  ← "Dairy-adjacent, breakfast"
    Cheese:  [0.51, 0.15, -0.41, ...]  ← "Dairy, premium"
    Apples:  [0.22, 0.33, -0.11, ...]  ← "Fresh, healthy, seasonal"
    Chicken: [0.18, 0.44, -0.22, ...]  ← "Meat, protein, dinner"

T3: Temporal Context [64 numbers]
    [0.67, -0.23, 0.45, 0.89, ...] ← Encodes: "Week 11 of year,
                                               Tuesday,
                                               10:30 AM,
                                               Spring"

T4: Price Features [6 products × 64 numbers]
    Milk:    [0.12, 0.34, -0.56, ...]  ← "Normal price, stable"
    Bread:   [0.08, 0.41, -0.33, ...]  ← "Normal price"
    Eggs:    [-0.23, 0.55, -0.12, ...] ← "Slightly cheap today"
    Cheese:  [0.45, 0.22, -0.67, ...]  ← "On promotion!"
    Apples:  [0.33, 0.18, -0.44, ...]  ← "In season, good price"
    Chicken: [0.67, 0.11, -0.89, ...]  ← "Premium, full price"

T5: Store Context [96 numbers]
    [0.34, 0.56, -0.12, 0.78, ...] ← Encodes: "Large supermarket,
                                               suburban,
                                               family area,
                                               full range"

T6: Trip Context [48 numbers]
    [0.89, -0.45, 0.23, 0.67, ...] ← Encodes: "Full shop mission,
                                               mixed focus,
                                               mainstream price sensitivity,
                                               medium basket"
```

---

### Masked Language Modeling: How the Model Learns

```
THE TRAINING GAME: "Fill in the Blanks"
══════════════════════════════════════════════════════════════════════════

Original basket:
    [Milk] [Bread] [Eggs] [Cheese] [Apples] [Chicken]
      1       2      3       4        5         6

Step 1: Randomly mask 15% of products
────────────────────────────────────────

    [Milk] [MASK] [Eggs] [Cheese] [MASK] [Chicken]
      1       2      3       4        5         6
            ↑                       ↑
         Hidden!                 Hidden!

Step 2: Model sees the context + masked basket
────────────────────────────────────────

    Given:
    • Customer: Family shopper, loves dairy, weekly visitor
    • Time: Tuesday morning, Spring
    • Store: Large suburban supermarket
    • Trip: Full shop, mixed focus
    • Visible products: Milk, Eggs, Cheese, Chicken
    • Hidden positions: 2 and 5

Step 3: Model predicts what's hidden
────────────────────────────────────────

    For position 2:
    "Based on everything I know, position 2 is probably..."

    Top predictions:
    1. Bread (0.73)  ← CORRECT! Model learned!
    2. Butter (0.12)
    3. Cereal (0.08)
    4. Jam (0.04)
    ...

    For position 5:
    "Based on everything I know, position 5 is probably..."

    Top predictions:
    1. Bananas (0.45)
    2. Oranges (0.22)
    3. Apples (0.18)  ← CORRECT! Close!
    4. Grapes (0.08)
    ...

Step 4: Calculate loss and update weights
────────────────────────────────────────

    Loss = How wrong were the predictions?

    Position 2: Predicted Bread (73%), actual was Bread → LOW loss
    Position 5: Predicted Bananas (45%), actual was Apples → MEDIUM loss

    Backpropagate → Adjust model weights → Better next time!
```

---

## The Training Process Step by Step

### Three Training Phases

```
TRAINING PHASES: 20 Epochs Total
══════════════════════════════════════════════════════════════════════════

PHASE 1: WARM-UP (Epochs 1-3)
──────────────────────────────────────────────────────────────────────────
Goal: Get the model started without overwhelming it

Settings:
• Learning rate: 0.00001 (very small steps)
• Mask rate: 15% (standard)
• Loss: Only product prediction (focal loss)
• Auxiliary tasks: OFF

What happens:
  The model learns basic patterns like:
  "Milk often appears with bread"
  "Saturday baskets are bigger"
  "Premium customers buy premium products"

PHASE 2: MAIN TRAINING (Epochs 4-15)
──────────────────────────────────────────────────────────────────────────
Goal: Full learning with all objectives

Settings:
• Learning rate: 0.00005 (peak learning)
• Mask rate: 15%
• Loss: Multi-task (all components active)
  - 60% Focal loss (product prediction)
  - 20% Contrastive loss (product relationships)
  - 20% Auxiliary losses (basket size, price sensitivity, etc.)

What happens:
  The model learns nuanced patterns:
  "Customer 12345 on Tuesday morning in Store 42
   doing a full shop will likely buy..."
  "These products are substitutes for each other"
  "This is a price-sensitive shopping trip"

PHASE 3: FINE-TUNING (Epochs 16-20)
──────────────────────────────────────────────────────────────────────────
Goal: Polish and refine on harder examples

Settings:
• Learning rate: 0.00001 (small steps again)
• Mask rate: 20% (harder - more hidden)
• Loss: Same as Phase 2
• Focus: Validation performance

What happens:
  The model refines edge cases:
  "Even when I hide more products, I can still predict"
  "I should be more confident in my top predictions"
  "Let me focus on the cases I'm getting wrong"
```

### Loss Functions Explained Simply

```
LOSS FUNCTIONS: How We Measure "Wrong"
══════════════════════════════════════════════════════════════════════════

1. FOCAL LOSS (60% weight)
────────────────────────────────────────
"How wrong are my product predictions?"

Problem it solves:
  We have 5000+ products but only ~10 in each basket.
  Regular loss would just predict "don't buy" for everything.

Focal loss solution:
  Punishes confident wrong answers HEAVILY.
  Ignores easy correct answers ("don't buy obscure products").
  Focuses on the hard predictions that matter.

Example:
  Model says: "Definitely NOT buying caviar" → Easy, ignore
  Model says: "Probably NOT buying milk" → Wrong! Big penalty!


2. CONTRASTIVE LOSS (20% weight)
────────────────────────────────────────
"Do I understand which products go together?"

What it does:
  Teaches the model that co-purchased products should be SIMILAR.
  Products from different baskets should be DIFFERENT.

Example:
  Same basket: [Pasta, Pasta Sauce] → Should be similar in model's view
  Different baskets: [Pasta] vs [Nappies] → Should be different

Why it helps:
  If you're predicting products for a pasta dinner, the model knows
  to suggest things that "look like" pasta in its internal representation.


3. AUXILIARY LOSSES (20% weight combined)
────────────────────────────────────────
"Do I understand the context of this shopping trip?"

Sub-tasks:
• Basket Size (8%): "Is this a small, medium, or large shop?"
• Price Sensitivity (8%): "Is this customer price-conscious?"
• Mission Type (4%): "Is this a full shop or quick top-up?"

Why these help:
  Forces the model to really understand the CONTEXT.
  A model that knows "this is a big weekly shop" will make
  better predictions than one that just sees products.
```

---

## Why Each Piece Matters

### Removing Components: What Breaks?

```
ABLATION STUDY: What if we remove each tensor?
══════════════════════════════════════════════════════════════════════════

WITHOUT T1 (Customer Context):
────────────────────────────────────────
Result: -15% accuracy

What breaks:
  Model treats all customers the same!
  Can't distinguish between:
  • Family vs. single shopper
  • Loyal customer vs. new visitor
  • Dairy lover vs. vegan

  Predictions become "average" and generic.


WITHOUT T2 (Product Embeddings):
────────────────────────────────────────
Result: -40% accuracy (catastrophic)

What breaks:
  Model doesn't understand products at all!
  Can't learn that:
  • Milk and bread go together
  • Coke and Pepsi are substitutes
  • Organic apples are "similar" to regular apples

  Predictions become random.


WITHOUT T3 (Temporal Context):
────────────────────────────────────────
Result: -12% accuracy

What breaks:
  Model ignores time patterns!
  Can't distinguish between:
  • Saturday big shop vs. Monday quick trip
  • Morning coffee run vs. evening dinner prep
  • Christmas shopping vs. regular week

  Predictions miss seasonal/time patterns.


WITHOUT T4 (Price Features):
────────────────────────────────────────
Result: -8% accuracy

What breaks:
  Model ignores price signals!
  Can't respond to:
  • Promotions and discounts
  • Price-sensitive customer behavior
  • Stock-up opportunities

  Predictions ignore economic behavior.


WITHOUT T5 (Store Context):
────────────────────────────────────────
Result: -6% accuracy

What breaks:
  Model treats all stores the same!
  Can't adapt to:
  • Small express store (limited range)
  • Large hypermarket (full selection)
  • Local demographics

  Predictions suggest unavailable products.


WITHOUT T6 (Trip Context):
────────────────────────────────────────
Result: -10% accuracy

What breaks:
  Model ignores shopping mission!
  Can't distinguish between:
  • Quick top-up (3 items)
  • Full weekly shop (40 items)
  • Special occasion (premium items)

  Predictions wrong in size and scope.
```

### The Synergy Effect

```
WHY ALL TENSORS TOGETHER > SUM OF PARTS
══════════════════════════════════════════════════════════════════════════

Individual understanding:
  T1 knows: "Customer likes dairy"
  T3 knows: "It's Saturday morning"
  T5 knows: "Large store with full range"
  T6 knows: "This is a full shop"

Combined understanding:
  "This dairy-loving customer is doing their Saturday full shop
   at a large store that stocks everything."

  → Predict: Full dairy section coverage
             (milk, cheese, yogurt, butter, cream)
  → Predict: Related breakfast items
             (eggs, bacon, orange juice)
  → Predict: Larger pack sizes
             (it's a weekly stock-up)

The model learns INTERACTIONS between tensors that wouldn't
be possible with any single piece of information.
```

---

## Quick Reference Card

```
TENSOR QUICK REFERENCE
══════════════════════════════════════════════════════════════════════════

TENSOR    DIM    WHAT IT CAPTURES                  KEY IMPACT
────────────────────────────────────────────────────────────────────────

T1       192    Customer identity & history       WHO is shopping
                • Shopping patterns               • Personalization
                • Category preferences            • Historical patterns
                • Demographics                    • Loyalty signals

T2       256    Product characteristics           WHAT products mean
        /prod   • Category membership             • Co-purchase patterns
                • Brand positioning               • Substitution logic
                • Co-purchase neighbors           • Product similarity

T3        64    Time context                      WHEN shopping happens
                • Day of week                     • Weekly patterns
                • Hour of day                     • Daily patterns
                • Season and holidays             • Seasonal items

T4        64    Price signals                     ECONOMIC context
        /prod   • Current vs. historical price    • Promotion response
                • Promotional indicators          • Price sensitivity
                • Volatility patterns             • Budget signals

T5        96    Store characteristics             WHERE shopping happens
                • Store format                    • Available products
                • Location demographics           • Local preferences
                • Product assortment              • Store-specific patterns

T6        48    Shopping mission                  WHY shopping happens
                • Trip type                       • Basket size expectations
                • Mission focus                   • Category focus
                • Price sensitivity               • Item count predictions
```

---

## Running the Training Pipeline

Now that you understand how everything works, here's how to actually run training.

### Prerequisites

Before training, you need to run the earlier pipeline stages. Here's the complete workflow:

```
COMPLETE WORKFLOW: From Raw Data to Trained Model
══════════════════════════════════════════════════════════════════════════

Step 0: Install Dependencies
─────────────────────────────

    # Using Poetry (recommended)
    poetry install

    # Or using pip
    pip install -e .

    Required packages:
    • torch >= 2.0.0
    • pandas >= 2.0.0
    • numpy >= 1.24.0
    • pyarrow >= 14.0.0


Step 1: Run the Data Pipeline (Section 2)
─────────────────────────────

    python -m src.data_pipeline.run_pipeline --nrows 10000

    What this does:
    • Stage 1: Derives unit prices from transactions
    • Stage 2: Builds the product graph (co-purchase relationships)
    • Stage 3: Calculates customer-store affinity
    • Stage 4: Extracts shopping mission patterns

    Output files (in data/processed/):
    • prices_derived.parquet
    • product_graph.pkl
    • customer_store_affinity.parquet
    • customer_mission_patterns.parquet

    Time: ~5 seconds for 10,000 transactions


Step 2: Run Feature Engineering (Section 3)
─────────────────────────────

    python -m src.feature_engineering.run_feature_engineering --nrows 10000

    What this does:
    • Layer 1: Creates pseudo-brand groupings
    • Layer 2: Generates Fourier-encoded price features
    • Layer 3: Trains product graph embeddings (GraphSAGE)
    • Layer 4: Builds customer history embeddings
    • Layer 5: Creates store context features

    Output files (in data/features/):
    • pseudo_brands.parquet
    • price_features.parquet
    • product_embeddings.pkl
    • customer_embeddings.parquet
    • store_features.parquet

    Time: ~60 seconds for 10,000 transactions


Step 3: Run Tensor Preparation (Section 4)
─────────────────────────────

    python -m src.tensor_preparation.run_tensor_preparation

    What this does:
    • Creates T1-T6 tensor builders
    • Validates all dimensions match spec
    • Builds dataset and dataloader

    This step validates everything is ready for training.

    Time: ~2 seconds
```

---

### Preparing Training Data

Training requires two preparation steps that organize the data for efficient training:

```
TRAINING DATA PREPARATION
══════════════════════════════════════════════════════════════════════════

Step 4: Prepare Training Samples
─────────────────────────────

    python -m src.training.prepare_samples

    What this does:
    • Loads all transactions and groups by basket
    • Adds temporal metadata (SHOP_WEEK, SHOP_WEEKDAY, SHOP_HOUR)
    • Splits into train/validation/test by week:
      - Train: Weeks 1-80
      - Validation: Weeks 81-95
      - Test: Weeks 96-117
    • Saves sample manifests

    Output files (in data/training/):
    • samples_train.parquet    (list of basket IDs for training)
    • samples_val.parquet      (list of basket IDs for validation)
    • samples_test.parquet     (list of basket IDs for testing)
    • temporal_metadata.parquet (week/day/hour for each basket)


Step 5: Prepare Tensor Cache
─────────────────────────────

    python -m src.training.prepare_tensor_cache

    What this does:
    • Pre-computes T1 (customer) embeddings for all customers
    • Pre-computes T5 (store) embeddings for all stores
    • Pre-computes T2 (product) embeddings for all products
    • Stores in memory-mapped files for fast loading

    Why caching matters:
    • Customer/store/product embeddings are STATIC
    • Computing them fresh every batch would be wasteful
    • Cache allows instant lookup: customer_id → [192 numbers]

    Output files (in data/training/cache/):
    • customer_embeddings.npy   (T1 cache)
    • store_embeddings.npy      (T5 cache)
    • product_embeddings.npy    (T2 cache)
    • index_mappings.pkl        (ID → array index)

    Memory: ~500MB for full dataset
```

---

### Running Training

Now you're ready to train the World Model:

```
TRAINING THE MODEL
══════════════════════════════════════════════════════════════════════════

Basic Training Command
─────────────────────────────

    python -m src.training.train

    This uses default settings:
    • Batch size: 256
    • Epochs: 20
    • Learning rate: 5e-5
    • Device: auto-detected (CUDA > MPS > CPU)


Training with Custom Settings
─────────────────────────────

    python -m src.training.train \
        --epochs 20 \
        --batch-size 256 \
        --learning-rate 5e-5 \
        --warmup-epochs 3 \
        --finetune-epochs 5 \
        --mask-prob 0.15

    Key parameters explained:

    --epochs 20
        Total training epochs (recommended: 20-30)

    --batch-size 256
        Samples per batch. Larger = faster but more memory.
        • GPU with 8GB: 128
        • GPU with 16GB: 256
        • GPU with 24GB+: 512

    --learning-rate 5e-5
        Peak learning rate during main phase.
        Warmup uses 1e-5, finetune uses 1e-5.

    --warmup-epochs 3
        Epochs at low learning rate to stabilize initial training.

    --finetune-epochs 5
        Final epochs with harder masking (20% instead of 15%).

    --mask-prob 0.15
        Fraction of products to mask (15% is standard BERT-style).


Resuming Training from Checkpoint
─────────────────────────────

    python -m src.training.train --resume checkpoints/epoch_10.pt

    Training saves checkpoints after each epoch:
    • checkpoints/epoch_1.pt
    • checkpoints/epoch_2.pt
    • ...
    • checkpoints/best_model.pt (lowest validation loss)


What You'll See During Training
─────────────────────────────

    Epoch 1/20 [Warmup]
    ────────────────────────────────────────────────
    Train Loss: 8.234  |  Val Loss: 7.891  |  Val Acc: 0.0312
    Phase: warmup  |  LR: 1e-5  |  Mask: 15%

    Epoch 2/20 [Warmup]
    ────────────────────────────────────────────────
    Train Loss: 6.127  |  Val Loss: 5.892  |  Val Acc: 0.0524
    Phase: warmup  |  LR: 1e-5  |  Mask: 15%

    ...

    Epoch 10/20 [Main]
    ────────────────────────────────────────────────
    Train Loss: 2.341  |  Val Loss: 2.567  |  Val Acc: 0.2341
    Phase: main  |  LR: 5e-5  |  Mask: 15%
    ✓ New best model saved!

    ...

    Epoch 20/20 [Finetune]
    ────────────────────────────────────────────────
    Train Loss: 1.892  |  Val Loss: 2.123  |  Val Acc: 0.2876
    Phase: finetune  |  LR: 1e-5  |  Mask: 20%

    Training complete!
    Best model: checkpoints/best_model.pt (epoch 18)
```

---

### Evaluating the Model

After training, evaluate on the held-out test set:

```
MODEL EVALUATION
══════════════════════════════════════════════════════════════════════════

Basic Evaluation
─────────────────────────────

    python -m src.training.evaluate checkpoints/best_model.pt

    This evaluates on the test set (weeks 96-117).


Evaluation on Different Splits
─────────────────────────────

    # Evaluate on validation set
    python -m src.training.evaluate checkpoints/best_model.pt --split val

    # Evaluate on test set (default)
    python -m src.training.evaluate checkpoints/best_model.pt --split test


Detailed Evaluation Output
─────────────────────────────

    python -m src.training.evaluate checkpoints/best_model.pt --detailed

    Output:
    ────────────────────────────────────────────────
    Overall Metrics
    ────────────────────────────────────────────────
    Top-1 Accuracy:     28.76%
    Top-5 Accuracy:     52.34%
    Top-10 Accuracy:    64.12%
    MRR (Mean Recip Rank): 0.3891

    ────────────────────────────────────────────────
    Breakdown by Day of Week
    ────────────────────────────────────────────────
    Monday:    Top-1: 26.3%  |  Top-5: 49.8%
    Tuesday:   Top-1: 27.1%  |  Top-5: 51.2%
    Wednesday: Top-1: 28.2%  |  Top-5: 52.0%
    Thursday:  Top-1: 28.9%  |  Top-5: 52.8%
    Friday:    Top-1: 29.4%  |  Top-5: 53.6%
    Saturday:  Top-1: 31.2%  |  Top-5: 55.1%  ← Best (more data)
    Sunday:    Top-1: 30.1%  |  Top-5: 53.9%

    ────────────────────────────────────────────────
    Breakdown by Basket Size
    ────────────────────────────────────────────────
    Small (1-5):     Top-1: 32.1%  |  Top-5: 58.3%
    Medium (6-15):   Top-1: 28.4%  |  Top-5: 51.9%
    Large (16+):     Top-1: 24.8%  |  Top-5: 47.2%

    ────────────────────────────────────────────────
    Breakdown by Customer History
    ────────────────────────────────────────────────
    New (< 5 trips):       Top-1: 18.2%  |  Top-5: 38.1%
    Regular (5-20 trips):  Top-1: 27.9%  |  Top-5: 51.8%
    Loyal (20+ trips):     Top-1: 34.2%  |  Top-5: 59.7%  ← Best


Understanding the Metrics
─────────────────────────────

    Top-1 Accuracy:
      "Did the model's #1 guess match the actual product?"
      28% means: 1 in 4 predictions exactly correct.
      (Random would be ~0.02% with 5000 products!)

    Top-5 Accuracy:
      "Was the actual product in the model's top 5 guesses?"
      52% means: More than half the time, correct answer in top 5.

    Top-10 Accuracy:
      "Was the actual product in the model's top 10 guesses?"
      64% means: For 2/3 of predictions, answer is in top 10.

    MRR (Mean Reciprocal Rank):
      Average of 1/rank for each prediction.
      If correct answer is rank 1: score = 1.0
      If correct answer is rank 2: score = 0.5
      If correct answer is rank 10: score = 0.1
      MRR of 0.39 means typical rank is ~2.5.
```

---

### Complete Training Recipe

Here's a copy-paste ready script to run the full pipeline:

```bash
# ═══════════════════════════════════════════════════════════════════════
# COMPLETE TRAINING RECIPE
# ═══════════════════════════════════════════════════════════════════════

# 0. Navigate to project
cd /path/to/retail_sim

# 1. Data Pipeline (5 seconds)
python -m src.data_pipeline.run_pipeline --nrows 10000

# 2. Feature Engineering (60 seconds)
python -m src.feature_engineering.run_feature_engineering --nrows 10000

# 3. Tensor Preparation (2 seconds)
python -m src.tensor_preparation.run_tensor_preparation

# 4. Prepare Training Samples
python -m src.training.prepare_samples

# 5. Prepare Tensor Cache
python -m src.training.prepare_tensor_cache

# 6. Train! (varies by hardware)
python -m src.training.train --epochs 20 --batch-size 256

# 7. Evaluate
python -m src.training.evaluate checkpoints/best_model.pt --detailed
```

---

### Troubleshooting Common Issues

```
TROUBLESHOOTING
══════════════════════════════════════════════════════════════════════════

ISSUE: Out of Memory (OOM)
─────────────────────────────
Error: CUDA out of memory

Solution: Reduce batch size
    python -m src.training.train --batch-size 64

Or enable gradient checkpointing:
    python -m src.training.train --gradient-checkpointing


ISSUE: Slow Training
─────────────────────────────
Training is taking forever!

Checks:
1. Are you using GPU?
   python -c "import torch; print(torch.cuda.is_available())"

2. For Mac: Are you using MPS?
   python -c "import torch; print(torch.backends.mps.is_available())"

3. Is data loading the bottleneck?
   Add: --num-workers 4


ISSUE: Loss Not Decreasing
─────────────────────────────
Loss stays flat or increases

Possible causes:
1. Learning rate too high
   Solution: --learning-rate 1e-5

2. Corrupted data
   Solution: Re-run data pipeline

3. Batch size too small
   Solution: --batch-size 128


ISSUE: NaN Loss
─────────────────────────────
Loss becomes NaN during training

Solutions:
1. Lower learning rate:
   --learning-rate 1e-5

2. Enable gradient clipping (on by default):
   --max-grad-norm 1.0

3. Check for bad data:
   python -m src.training.validate_data
```

---

## Conclusion

The World Model learns to predict shopping behavior by understanding:

1. **The Customer** (T1): Who they are, what they've bought before
2. **The Products** (T2): What each product means and how they relate
3. **The Time** (T3): When the shopping happens and what that implies
4. **The Prices** (T4): Economic signals that influence decisions
5. **The Store** (T5): Where the shopping happens and what's available
6. **The Mission** (T6): Why the customer is shopping today

By combining all six perspectives, the model can make predictions like:

> "Customer #12345, a family shopper who loves dairy, is doing their Saturday morning full shop at Store #42, a large suburban supermarket. It's late February, so seasonal items are shifting toward spring. Given their history and this context, they will likely buy: milk (95%), bread (92%), eggs (88%), cheese (75%), apples (71%), chicken (68%), yogurt (62%), butter (55%)..."

This is the power of multi-modal temporal learning for retail prediction.
