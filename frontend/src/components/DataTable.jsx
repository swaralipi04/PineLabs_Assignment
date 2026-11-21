import { formatCell, isNumeric } from "./format.js";

export default function DataTable({ table, maxHeight }) {
  if (!table || !table.columns?.length) return <p className="muted small">No rows.</p>;
  return (
    <>
      <div className="table-wrap" style={maxHeight ? { maxHeight } : undefined}>
        <table className="dt">
          <thead>
            <tr>{table.columns.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {table.rows.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td key={j} className={isNumeric(v) ? "num" : ""}>{formatCell(table.columns[j], v)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.total_rows > table.rows.length && (
        <p className="muted small">Showing {table.rows.length} of {table.total_rows.toLocaleString()} rows.</p>
      )}
    </>
  );
}
