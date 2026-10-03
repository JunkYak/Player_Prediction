// ─── DRAFT RULES & FIXED SLOT MULTIPLIERS ──────────────────────────────────────
export const SLOT_MULTIPLIERS = [2.0, 1.8, 1.6, 1.4, 1.2];

/**
 * Calculates the projected score contribution for a player in a specific slot.
 * projected_slot_value = predicted_rating * slot_multiplier
 *
 * @param {Object|null} player - The drafted player object (or null if empty)
 * @param {number} slotIndex - Index of the slot (0 to 4)
 * @returns {number} Projected value or 0 if slot is empty
 */
export function getSlotProjectedValue(player, slotIndex) {
  if (!player || typeof player.predicted_rating !== 'number') {
    return 0;
  }
  const multiplier = SLOT_MULTIPLIERS[slotIndex] ?? 1.0;
  return player.predicted_rating * multiplier;
}

/**
 * Calculates the total projected score for the 5-slot lineup.
 *
 * @param {Array<Object|null>} slots - Array of 5 slots
 * @returns {number} Sum of all projected slot values
 */
export function getTotalProjectedScore(slots) {
  if (!Array.isArray(slots)) return 0;
  return slots.reduce((total, player, idx) => {
    return total + getSlotProjectedValue(player, idx);
  }, 0);
}
