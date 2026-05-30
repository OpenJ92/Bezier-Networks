# Bezier Networks: A Research Reconstruction

This project is probably not best understood as an attempt to invent a universally better neural
network layer. It is better understood as a way of placing a geometric prior on architecture
construction.

The central idea is simple:

```text
architecture = a smooth path through tensor-shape space
```

Instead of describing a neural network as an arbitrary list of layers, we describe it as a path from
one tensor shape to another. A Bezier curve gives that path a small number of meaningful handles:
the control points. Moving those control points changes the shape of the architecture family without
requiring us to hand-edit every layer.

That is the part worth preserving.

## What Is Interesting Here

The strongest version of this idea is not "Bezier networks are better than ordinary networks." That
claim is too broad, too vague, and probably not true in that form.

The stronger and more testable claim is:

```text
Bezier shape paths are a compact way to generate constrained architecture families.
```

This matters because neural architectures are usually both highly structured and awkward to search.
They are not just sequences of layers; they are compression paths, expansion paths, bottlenecks,
resolutions, channel schedules, and parameter budgets. A Bezier curve is a natural way to describe
those paths with smoothness, locality, and controllability.

For example, an autoencoder can be viewed as a path from input shape to latent shape and then back
again. The encoder contracts through shape space. The decoder expands. If those two paths are
constructed as related curves, then the architecture has a geometric symmetry that is easy to
visualize and manipulate.

That does not mean it will perform better. It means it gives us a coherent space to experiment in.

## Why It Might Be Useful

The value of Bezier architecture construction is likely to come from constraints rather than raw
expressive power.

Useful constraints include:

- Smooth compression and expansion paths.
- Explicit control over bottleneck geometry.
- Compact parameterization of whole architecture families.
- Symmetric or mirrored encoder/decoder construction.
- Parameter, FLOP, or memory budgets applied along a path.
- Truncation policies that stop or reshape a curve when it becomes invalid or too expensive.
- Visual explanations of how a model moves through tensor-shape space.

In this framing, the Bezier curve is not the model. It is the generator of a model family.

That distinction matters. The model is still an ordinary PyTorch module. The curve is the recipe for
constructing it.

## Where The Idea Is Weak

There are real risks.

Shape smoothness may not correlate with learning quality. A beautiful path through shape space may
produce a mediocre network. A Bezier curve can also become a decorative way to describe an arbitrary
architecture if the control points are unconstrained. In that case, the geometry adds complexity
without adding discipline.

The convolutional construction is another weak point. The early version tried to encode spatial
movement mostly through convolution and transposed convolution kernel sizes. That preserves the old
idea, but it is not a modern or principled compiler strategy. A better design should separate the
job into clearer steps:

```text
channel change -> projection
spatial change -> interpolation or adaptive pooling
feature mixing -> convolution
```

Without that separation, the generated networks can become arbitrary and hard to reason about.

There is also a danger that the metaphor outruns the experiment. "Architectures as curves" is an
appealing image, but it only becomes serious if it creates measurable, reproducible comparisons.

## A Better Formulation

The broad question:

```text
Can Bezier networks become a new deep learning paradigm?
```

is probably not the right question.

A better question is:

```text
Can Bezier shape paths generate useful autoencoder families under explicit parameter,
resolution, and truncation budgets?
```

That question is narrow enough to test.

It suggests a concrete pipeline:

```text
Bezier control points
-> continuous curve
-> quantized ShapePath
-> truncated or repaired ShapePath
-> LayerPlan
-> PyTorch module
```

The important research objects are not only the curves. They are the policies around the curves:

- How are continuous shapes rounded into integer tensor shapes?
- When should a path be truncated?
- When should a path be repaired?
- How should mixed spatial movement be split into simpler monotone segments?
- How should parameter count or memory budget constrain the path?

Those questions are more important than the initial layer implementation.

## A Sensible Next Experiment

The first serious experiment should be small.

Build a minimal planner:

```text
ShapePath
LayerPlan
QuantizationPolicy
TruncationPolicy
```

Then generate a family of small autoencoders. For each generated architecture, record:

- The Bezier control points.
- The sampled shape path.
- The parameter count.
- The activation-memory estimate.
- The reconstruction loss.
- The training stability.

Start with simple data. MNIST, Fashion-MNIST, CIFAR-10, or even synthetic image data would be
enough. The point is not to win a benchmark. The point is to see whether the geometry produces
coherent, comparable architecture families.

A useful result would not have to be spectacular. Even a modest result could justify the idea if the
curves give a clean way to navigate tradeoffs between compression, reconstruction quality, and model
size.

## The Working Hypothesis

The working hypothesis is:

```text
Bezier curves are useful architectural priors when the task benefits from controlled
compression, expansion, truncation, or symmetry.
```

That points naturally toward autoencoders, VAEs, architecture morphisms, and budget-constrained
model generation.

It does not point naturally toward replacing every hand-designed architecture. That would be the
wrong burden to put on the idea.

## Bottom Line

This is not a dumb idea. It is a fragile idea.

It becomes weak if treated as a grand claim about neural networks. It becomes interesting if treated
as a compact, geometric, testable way to generate architecture families.

The next step is not more metaphor. The next step is a planner, policies, and experiments.
