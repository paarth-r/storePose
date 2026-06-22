import type { PersonState } from "./types";

// COCO-17 keypoint indices.
// 0 nose · 1/2 eyes · 3/4 ears · 5/6 shoulders · 7/8 elbows · 9/10 wrists
// 11/12 hips · 13/14 knees · 15/16 ankles
//
// A deliberately spare skeleton: limbs + torso + a single neck line to the nose.
// Facial micro-edges are omitted so people read as people, not wireframes.
export const SKELETON_EDGES: [number, number][] = [
  [5, 7], [7, 9], // left arm
  [6, 8], [8, 10], // right arm
  [5, 6], // shoulders
  [5, 11], [6, 12], [11, 12], // torso
  [11, 13], [13, 15], // left leg
  [12, 14], [14, 16], // right leg
  [0, 5], [0, 6], // neck → shoulders
];

// Joints worth marking with a dot (skip dense facial points).
export const JOINT_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/** Stroke/accent color for each state — tuned for legibility over photographic video. */
export function stateColor(state: PersonState): string {
  switch (state) {
    case "serving":
      return "#34d399"; // emerald — at the Mashgin checkout
    case "serving_other":
      return "#60a5fa"; // blue — staffed lane
    case "waiting":
      return "#fbbf24"; // amber — in line
    case "candidate":
      return "#a78bfa"; // violet — joining
    default:
      return "#e6e8ee"; // neutral — tracked / passing
  }
}
