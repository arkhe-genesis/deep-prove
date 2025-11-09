use colored::Colorize;
use compact_str::CompactString;
use std::io::Write;

#[derive(Default)]
struct KeyMap<T: Clone> {
    choices: Vec<(CompactString, T)>,
    acc: CompactString,
}
impl<T: Clone> KeyMap<T> {
    fn from_choices<'a>(choices: impl Iterator<Item = &'a Entry<T>>) -> Self
    where
        T: 'a,
    {
        Self {
            choices: choices
                .into_iter()
                .map(|entry| (entry.chord.clone(), entry.payload.clone()))
                .collect(),
            acc: Default::default(),
        }
    }

    fn read_char(&mut self, c: char) -> anyhow::Result<Option<&T>> {
        self.acc.push(c);
        let candidates = self
            .choices
            .iter()
            .filter(|(chord, _)| chord.starts_with(self.acc.as_str()))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            anyhow::bail!("no valid candidate found")
        } else if candidates.len() == 1 {
            Ok(Some(&candidates[0].1))
        } else {
            Ok(None)
        }
    }

    fn clear(&mut self) {
        self.acc.clear();
    }
}

#[derive(Clone)]
pub struct Entry<T: Clone> {
    pub chord: CompactString,
    pub label: CompactString,
    pub payload: T,
    pub show: bool,
}

pub fn menu<T: Clone>(
    tty: &mut console::Term,
    title: &str,
    choices: &[Entry<T>],
) -> anyhow::Result<T> {
    let prompt = format!(
        "{} {}",
        title.white().bold(),
        choices
            .iter()
            .filter_map(|entry| if entry.show {
                Some(format!(
                    "{} {}",
                    format!("[{}]", entry.chord).yellow().bold(),
                    entry.label
                ))
            } else {
                None
            })
            .collect::<Vec<_>>()
            .join(" - "),
    );
    let mut key_map = KeyMap::from_choices(choices.iter());

    writeln!(tty, "{prompt}").unwrap();
    loop {
        match key_map.read_char(tty.read_char()?) {
            Ok(None) => {}
            Ok(Some(e)) => return Ok(e.clone()),
            Err(_) => {
                println!("Unknown key chord `{}`", key_map.acc.red());
                key_map.clear();
            }
        }
    }
}
