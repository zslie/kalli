use std::collections::HashMap;
use std::collections::HashSet;

pub struct WordFrequency {
    dictionary: HashMap<String, i32>,
    total_words: i32,
    unique_words: i32,
    letters: HashSet<char>,
    tokenizer: fn(String) -> Vec<String>, // Assuming tokenizer as a function pointer
    case_sensitive: bool,
    longest_word_length: usize,
}

impl WordFrequency {
    pub fn new(tokenizer: Option<fn(String) -> Vec<String>>, case_sensitive: bool) -> Self {
        let dictionary = HashMap::new();
        let total_words = 0;
        let unique_words = 0;
        let letters = HashSet::new();
        let longest_word_length = 0;

        let tokenizer = match tokenizer {
            Some(t) => t,
            None => _parse_into_words, // Assuming _parse_into_words as a default tokenizer
        };

        Self {
            dictionary,
            total_words,
            unique_words,
            letters,
            tokenizer,
            case_sensitive,
            longest_word_length,
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        let key = ensure_unicode(key.to_string());
        let key = if self.case_sensitive {
            key
        } else {
            key.to_lowercase()
        };
        self.dictionary.contains_key(&key)
    }

    pub fn get(&self, key: &str) -> Option<&i32> {
        let key = ensure_unicode(key.to_string());
        let key = if self.case_sensitive {
            key
        } else {
            key.to_lowercase()
        };
        self.dictionary.get(&key)
    }

    pub fn iter(&self) -> std::collections::hash_map::Keys<String, i32> {
        self.dictionary.keys()
    }

    pub fn pop(&mut self, key: &str, default: Option<i32>) -> i32 {
        let key = ensure_unicode(key.to_string());
        let key = if self.case_sensitive {
            key
        } else {
            key.to_lowercase()
        };
        match self.dictionary.remove(&key) {
            Some(value) => value,
            None => match default {
                Some(default_value) => default_value,
                None => panic!("Key not found and default not provided"), // You may want to handle this case differently
            },
        }
    }

    pub fn dictionary(&self) -> &HashMap<String, i32> {
        &self.dictionary
    }

    pub fn total_words(&self) -> i32 {
        self.total_words
    }

    fn unique_words(&self) -> i32 {
        self.unique_words
    }

    // letters
    fn letters(&self) -> HashSet<char> {
        self.letters.clone()
    }

    // longest_word_length
    fn longest_word_length(&self) -> i32 {
        self.longest_word_length
    }

    // tokenize
    fn tokenize(&self, text: String) -> Vec<String> {
        let tmp_text = ensure_unicode(text); // Assumes ensure_unicode is defined
        let words = (self.tokenizer)(tmp_text);
        if self.case_sensitive {
            words
        } else {
            words.iter().map(|word| word.to_lowercase()).collect()
        }
    }

    // keys
    fn keys(&self) -> Vec<String> {
        self.dictionary.keys().cloned().collect()
    }

    fn words(&self) -> Vec<String> {
        self.dictionary.keys().cloned().collect()
    }

    // items
    fn items(&self) -> Vec<(String, i32)> {
        self.dictionary
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    // load_dictionary
    fn load_dictionary(&mut self, filename: &str, encoding: &str) {
        // Assume load_file is a defined function
        let data = load_file(filename, encoding);

        // Check case sensitivity
        let data = if self.case_sensitive {
            data
        } else {
            data.to_lowercase()
        };

        // Assume JSON is parsed via serde_json crate
        let loaded_data: HashMap<String, i32> = serde_json::from_str(&data).unwrap();
        self.dictionary.extend(loaded_data);
        self.update_dictionary(); // Assume update_dictionary is a defined method
    }

    // load_json
    fn load_json(&mut self, data: HashMap<String, i32>) {
        self.dictionary.extend(data);
        self.update_dictionary();
    }

    // load_text_file
    fn load_text_file(
        &mut self,
        filename: &str,
        encoding: &str,
        tokenizer: Option<Box<dyn Fn(String) -> Vec<String>>>,
    ) {
        let data = load_file(filename, encoding);
        self.load_text(data, tokenizer);
    }

    // load_text
    fn load_text(&mut self, text: String, tokenizer: Option<Box<dyn Fn(String) -> Vec<String>>>) {
        let text = ensure_unicode(text); // Assumes ensure_unicode is defined

        let words = match tokenizer {
            Some(tokenizer_fn) => {
                let words = tokenizer_fn(text);
                if self.case_sensitive {
                    words
                } else {
                    words.iter().map(|word| word.to_lowercase()).collect()
                }
            }
            None => self.tokenize(text),
        };

        for word in words {
            let counter = self.dictionary.entry(word).or_insert(0);
            *counter += 1;
        }

        self.update_dictionary();
    }

    // load_words
    fn load_words(&mut self, words: Vec<String>) {
        let words = words
            .iter()
            .map(|w| ensure_unicode(w.clone()))
            .collect::<Vec<String>>();
        for word in words {
            let word = if self.case_sensitive {
                word
            } else {
                word.to_lowercase()
            };
            self.dictionary.entry(word).or_insert(0);
        }
        self.update_dictionary();
    }

    // add
    fn add(&mut self, word: String, val: i32) {
        let word = ensure_unicode(word);
        let word = if self.case_sensitive {
            word
        } else {
            word.to_lowercase()
        };
        self.load_json(
            vec![(word, val)]
                .into_iter()
                .collect::<HashMap<String, i32>>(),
        );
    }

    // remove_words
    fn remove_words(&mut self, words: Vec<String>) {
        let words = words
            .iter()
            .map(|w| ensure_unicode(w.clone()))
            .collect::<Vec<String>>();
        for word in words {
            self.pop(&word, 0.0);
        }
        self.update_dictionary();
    }

    // remove
    fn remove(&mut self, word: String) {
        self.pop(&word, 0.0);
        self.update_dictionary();
    }

    // remove_by_threshold
    fn remove_by_threshold(&mut self, threshold: i32) {
        let to_remove = self
            .dictionary
            .iter()
            .filter(|&(_, &v)| v <= threshold)
            .map(|(&k, _)| k.clone())
            .collect::<Vec<String>>();
        self.remove_words(to_remove);
    }

    // update_dictionary
    fn update_dictionary(&mut self) {
        self.longest_word_length = 0;
        self.total_words = self.dictionary.values().sum();
        self.unique_words = self.dictionary.len();
        self.letters.clear();
        for key in self.dictionary.keys() {
            if key.len() > self.longest_word_length {
                self.longest_word_length = key.len();
            }
            self.letters.extend(key.chars());
        }
    }
}
