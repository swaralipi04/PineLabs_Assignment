// Column-aware value formatting for result tables.

const PCT = /(rate|share)/i;
const PTS = /(effect|contribution)$/i;

export function formatCell(col, v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (PTS.test(col)) return `${(v * 100).toFixed(2)} pts`;
    if (PCT.test(col) && Math.abs(v) <= 1.5) return `${(v * 100).toFixed(2)}%`;
    if (/(amount|gmv|inr|fees)/i.test(col)) return `₹${v.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
    if (Number.isInteger(v)) return v.toLocaleString("en-IN");
    return v.toLocaleString("en-IN", { maximumFractionDigits: 2 });
  }
  return String(v);
}

export const isNumeric = (v) => typeof v === "number";
