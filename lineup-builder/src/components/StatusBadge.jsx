// Status badge component
export default function StatusBadge({ status }) {
  const safeStatus = status || 'Active';
  const cls = {
    Active: 'status-active',
    Probable: 'status-probable',
    Questionable: 'status-questionable',
    Out: 'status-out',
  }[safeStatus] || 'status-active';

  return (
    <span className={`status-badge ${cls}`}>
      {safeStatus}
    </span>
  );
}
