# Study summary: Linear Algebra Intuition (Phase 1, Lesson 1)

## Vector
- A list of numbers, such as `[3, 2]`. It's a point or a direction (3 right, 2 up).
- In AI: a word, an image or a user becomes a vector.

## Dot product
- Rule: multiply the matching pairs, then add.
- `[1, 2] · [3, 4] = 1*3 + 2*4 = 11`
- Sign tells you how the vectors relate:
  - positive: similar direction
  - zero: perpendicular, unrelated
  - negative: opposite direction
- Watch the signs: `1 * (-2) = -2`.
- Used in search, RAG and attention.

## Matrix
- A grid of numbers that acts as a transformation: a vector goes in and a new vector comes out.
- Each output entry is **one row of the matrix dotted with the input**.
- `[[1, 2], [3, 4]] @ [2, 1] = [1*2+2*1, 3*2+4*1] = [4, 10]`
- Output size = number of rows. A 2x3 matrix takes 3 numbers in and gives 2 out.
- A neural network layer is exactly this operation.

## Examples of transformations
- Scaling: `[[3, 0], [0, 2]]` triples x and doubles y.
- 90 degree rotation: `[[0, -1], [1, 0]]` turns `[1, 0]` into `[0, 1]` and `[0, 1]` into `[-1, 0]`.

## Linear independence
- A set is independent if no vector can be built from the others.
- `[2, 1] = 2*[1, 0] + 1*[0, 1]`, so it is dependent on those two.
- A dependent feature adds no new information for a model.

## Rank
- The number of truly independent columns of a matrix.
- Low rank means redundant information. LoRA is built on this idea.

## Projection
- The "shadow" of one vector on another. Projecting `[3, 4]` onto `[1, 0]` gives `[3, 0]`.
- Used in regression and PCA.

## Code
- `zip` plus `sum` gives the dot product.
- NumPy's `@` does matrix multiplication without the loops.

## Where I stumbled (review)
- Multiplying with negative numbers.
- Matrix output size (rows, not total entries).
