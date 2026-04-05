// Status badge component
export default function StatusBadge({ status }) {
  const cls = {
    Active: 'status-active',
    Probable: 'status-probable',
    Questionable: 'status-questionable',
    Out: 'status-out',
  }[status] || 'status-active';

  return (
    <span className={`status-badge ${cls}`}>
      {status}
    </span>
  );
}
