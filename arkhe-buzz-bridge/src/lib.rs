// Arkhe Buzz Bridge - Nostr Integration (v0.31 compatible)

use nostr_sdk::prelude::*;

pub struct BuzzBridge {
    keys: Keys,
}

impl BuzzBridge {
    pub fn new(secret_key: &str) -> Result<Self, nostr_sdk::event::unsigned::Error> {
        let keys = Keys::parse(secret_key).unwrap_or_else(|_| Keys::generate());
        Ok(Self { keys })
    }

    pub fn publish_experiment(&self, experiment_id: String, content: String) -> Result<Event, nostr_sdk::event::builder::Error> {
        // v0.31 compatible tag creation and event building
        let tags = vec![Tag::custom(TagKind::from("experiment"), vec![experiment_id])];
        EventBuilder::new(Kind::Custom(30000), content, tags).to_event(&self.keys)
    }

    pub fn firewall_allows_edge(zone_from: &str, zone_to: &str, is_translation: bool) -> bool {
        if zone_from == "Z0" && zone_to == "Z2" {
            return is_translation;
        }
        false
    }
}
