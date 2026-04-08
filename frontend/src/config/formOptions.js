/**
 * FORM_OPTIONS — single source of truth for all editorial field option sets.
 * Values MUST match the Literal types in api/models.py EditorialInput.
 * If the API contract changes, update this file only.
 */

const FORM_OPTIONS = {
  category: [
    { value: 'Economy',  label: 'Economy',  helper: 'Financial markets and economic indicators' },
    { value: 'Politics', label: 'Politics', helper: 'Government, elections, and policy' },
    { value: 'Tech',     label: 'Tech',     helper: 'Technology, innovation, and science' },
    { value: 'Climate',  label: 'Climate',  helper: 'Environment, weather, and climate change' },
    { value: 'Crime',    label: 'Crime',    helper: 'Law enforcement and criminal justice' },
    { value: 'Sports',   label: 'Sports',   helper: 'Athletics, competition, and sports news' },
    { value: 'General',  label: 'General',  helper: 'General news and current affairs' },
  ],

  location_feel: [
    { value: 'Urban',      label: 'Urban',      helper: 'City streets, skylines, and built environment' },
    { value: 'Government', label: 'Government', helper: 'Institutional buildings and civic spaces' },
    { value: 'Nature',     label: 'Nature',     helper: 'Landscapes, outdoors, and natural settings' },
    { value: 'Abstract',   label: 'Abstract',   helper: 'Conceptual, non-literal visual space' },
    { value: 'Data',       label: 'Data',       helper: 'Charts, grids, and data-driven imagery' },
  ],

  time_of_day: [
    { value: 'Day',   label: 'Day',   helper: 'Full daylight, open and bright' },
    { value: 'Dusk',  label: 'Dusk',  helper: 'Golden hour, transitional light' },
    { value: 'Night', label: 'Night', helper: 'Artificial light, low ambient' },
    { value: 'N/A',   label: 'N/A',   helper: 'Time-neutral or abstract setting' },
  ],

  color_temperature: [
    { value: 'Neutral', label: 'Neutral', helper: 'No grade. Clean starting point.' },
    { value: 'Cool',    label: 'Cool',    helper: 'Desaturated blue-shifted. Authority look.' },
    { value: 'Warm',    label: 'Warm',    helper: 'Slightly warm, slightly crushed. Urgency.' },
  ],

  mood: [
    { value: 'Serious',   label: 'Serious',   helper: 'Grave, weighty, authoritative tone' },
    { value: 'Tense',     label: 'Tense',     helper: 'Heightened stakes, unresolved conflict' },
    { value: 'Neutral',   label: 'Neutral',   helper: 'Balanced, informational, uncolored' },
    { value: 'Uplifting', label: 'Uplifting', helper: 'Positive, forward-looking, energetic' },
  ],

  motion_intensity: [
    { value: 'Minimal', label: 'Minimal', helper: 'Near-static, subtle drift or hold' },
    { value: 'Gentle',  label: 'Gentle',  helper: 'Slow pan or gentle environmental motion' },
    { value: 'Dynamic', label: 'Dynamic', helper: 'Active movement, noticeable kinetic energy' },
  ],
}

export default FORM_OPTIONS
