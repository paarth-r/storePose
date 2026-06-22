// Shapes mirroring the storePose dashboard server payloads.

export type PersonState =
  | "tracked"
  | "candidate"
  | "waiting"
  | "serving"
  | "serving_other"
  | "out";

export interface OverlayPerson {
  id: number;
  box: [number, number, number, number]; // x1,y1,x2,y2 in original frame coords
  kpts: [number, number, number][]; // [x, y, score] * 17 (COCO), [] while coasting
  state: PersonState;
  wait: number;
  serve: number;
  progress: number;
}

export interface OverlayZones {
  line?: number[][][];
  pos?: number[][][];
  alt?: number[][][];
}

export interface StreamEvent {
  seq: number;
  t?: number;
  w: number;
  h: number;
  jpeg: string; // data:image/jpeg;base64,...
  people: OverlayPerson[];
  zones: OverlayZones;
  busy: { level: string | null; value: number } | null;
}

export interface Metrics {
  now: number;
  summary: {
    in_line: number;
    at_pos: number;
    avg_line_s: number;
    avg_pos_s: number;
    avg_total_s: number;
    served_count: number;
  };
  busy: {
    current: { level: string | null; value: number };
    t: number[];
    level_idx: number[];
    value: number[];
  };
  checkouts: {
    mashgin_avg: number;
    mashgin_avg_eff: number;
    num_mashgins: number;
    mashgin_n: number;
    other_avg: number;
    other_n: number;
    delta: number;
    series: {
      t_mashgin: number[];
      mashgin_ma: number[];
      t_other: number[];
      other_ma: number[];
    };
  };
  occupancy: {
    t: number[];
    waiting: number[];
    serving: number[];
    waiting_ma: number[];
    serving_ma: number[];
  };
  wait_serve: { t: number[]; wait_ma: number[]; serve_ma: number[] };
  throughput: { t: number[]; served_per_min: number[] };
  debug: { frame: number | null; rows: DebugRow[] };
}

export interface DebugRow {
  id: number;
  state: string;
  wait: number;
  serve: number;
  speed: number;
  line: boolean;
  pos: boolean;
  reg: boolean;
  transit: boolean;
}
