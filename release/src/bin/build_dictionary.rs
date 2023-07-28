use clap::{App, Arg};
use rust_bert::pipelines::tokenization::{BertTokenizer, BertVocab, Tokenizer};
use rust_stemmers::{Algorithm, Stemmer};
use serde_json::json;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

const MINIMUM_FREQUENCY: i32 = 50;
const LANGUAGES: [&str; 9] = ["en", "es", "de", "fr", "pt", "ru", "ar", "lv", "eu"];

fn load_file(filename: &str, encoding: &str) -> io::Result<String> {
    if filename.ends_with(".gz") {
        let file = File::open(filename)?;
        let mut gz = flate2::read::GzDecoder::new(file);
        let mut s = String::new();
        gz.read_to_string(&mut s)?;
        Ok(s)
    } else {
        let file = File::open(filename)?;
        let mut buf_reader = BufReader::new(file);
        let mut contents = String::new();
        buf_reader.read_to_string(&mut contents)?;
        Ok(contents)
    }
}

fn export_word_frequency(
    filepath: &str,
    word_frequency: &HashMap<String, i32>,
) -> std::io::Result<()> {
    let file = File::create(filepath)?;
    let mut file = std::io::BufWriter::new(file);
    let j = json!(word_frequency);
    let serialized = serde_json::to_string_pretty(&j).unwrap();
    write!(file, "{}", serialized)?;
    Ok(())
}

fn build_word_frequency(
    filepath: &str,
    language: &str,
    output_path: &str,
) -> std::io::Result<HashMap<String, i32>> {
    let mut word_frequency = HashMap::new();
    let tokenizer: BertTokenizer;

    if language == "es" {
        tokenizer = BertTokenizer::from_file("bert-base-multilingual-uncased", false);
    } else {
        tokenizer = BertTokenizer::from_file("bert-base-uncased", false);
    }

    let stemmer = Stemmer::create(Algorithm::English);
    let mut idx = 0;

    let input = File::open(filepath)?;
    let buffered = std::io::BufReader::new(input);

    for line in buffered.lines() {
        let l = line?;
        let tokenized_input = tokenizer.tokenize(&l);
        let tokens: Vec<&str> = tokenized_input.iter().map(|t| t.as_str()).collect();

        let mut words = vec![];

        for word in &tokens {
            if word.chars().all(|c| c.is_alphabetic()) && !stemmer.stem(word).is_empty() {
                words.push(word.to_string());
            }
        }

        if !words.is_empty() {
            for word in words {
                *word_frequency.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        idx += 1;

        if idx % 100000 == 0 {
            println!("completed: {} rows", idx);
        }
    }

    println!("completed: {} rows", idx);
    export_word_frequency(output_path, &word_frequency)?;

    Ok(word_frequency)
}

fn export_misfit_words(
    misfit_filepath: &str,
    word_freq_filepath: &str,
    word_frequency: &HashMap<String, i32>,
) -> std::io::Result<()> {
    let file = File::open(word_freq_filepath)?;
    let source_word_frequency: HashMap<String, Value> = serde_json::from_reader(file)?;

    let source_words: std::collections::HashSet<_> =
        source_word_frequency.keys().cloned().collect();
    let final_words: std::collections::HashSet<_> = word_frequency.keys().cloned().collect();

    let misfitted_words: Vec<_> = source_words.difference(&final_words).cloned().collect();
    let mut misfitted_words: Vec<String> = misfitted_words.iter().cloned().collect();
    misfitted_words.sort();

    let file = File::create(misfit_filepath)?;
    let mut file = BufWriter::new(file);
    for word in misfitted_words {
        writeln!(file, "{}", word)?;
    }
    Ok(())
}

fn clean_english(
    word_frequency: &HashMap<String, i32>,
    filepath_exclude: &str,
    filepath_include: &str,
) -> HashMap<String, i32> {
    let letters: std::collections::HashSet<char> = "abcdefghijklmnopqrstuvwxyz'".chars().collect();
    let vowels: std::collections::HashSet<char> = "aeiouy".chars().collect();

    let mut new_word_frequency = HashMap::new();

    for (key, &value) in word_frequency.iter() {
        let word_chars: std::collections::HashSet<_> = key.chars().collect();

        // Remove words with invalid characters, without a vowel, double punctuations, ellipses, and leading or trailing doubles
        if !word_chars.is_subset(&letters)
            || word_chars.is_disjoint(&vowels)
            || key.matches("'").count() > 1
            || key.matches("-").count() > 1
            || key.matches(".").count() > 2
            || key.contains("..")
            || (key.starts_with("aa") && !["aardvark", "aardvarks"].contains(&key.as_str()))
            || key.starts_with("a'")
            || key.starts_with("zz")
            || key.ends_with("yy")
            || key.ends_with("hh")
            || key.starts_with("about") && key != "about"
            || key.starts_with("above") && key != "above"
            || key.starts_with("after") && key != "after"
            || key.starts_with("against") && key != "against"
            || key.starts_with("all") && value < 15
            || key.starts_with("almost") && key != "almost"
            || key.starts_with("to") && value < 25
            || key.starts_with("can't") && key != "can't"
            || key.starts_with("i'm") && key != "i'm"
            || value <= MINIMUM_FREQUENCY
        {
            continue;
        }

        new_word_frequency.insert(key.clone(), value);
    }

    // Remove flagged misspellings
    let input = File::open(filepath_exclude).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        new_word_frequency.remove(l.trim());
    }

    // Add known missing words back in (ugh)
    let input = File::open(filepath_include).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        let l = l.trim();
        if new_word_frequency.contains_key(l) {
            println!("{} is already found in the dictionary! Skipping!", l);
        } else {
            new_word_frequency.insert(l.to_string(), MINIMUM_FREQUENCY);
        }
    }

    new_word_frequency
}

fn clean_german(
    word_frequency: &HashMap<String, i32>,
    filepath_exclude: &str,
    filepath_include: &str,
) -> HashMap<String, i32> {
    let letters: std::collections::HashSet<char> =
        "abcdefghijklmnopqrstuvwxyzäöüß".chars().collect();

    let mut new_word_frequency = HashMap::new();

    for (key, &value) in word_frequency.iter() {
        let word_chars: std::collections::HashSet<_> = key.chars().collect();

        // Remove words with invalid characters and words that start with a double "a"
        if !word_chars.is_subset(&letters) || key.starts_with("aa") || value <= MINIMUM_FREQUENCY {
            continue;
        }

        new_word_frequency.insert(key.clone(), value);
    }

    // Remove flagged misspellings
    let input = File::open(filepath_exclude).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        new_word_frequency.remove(l.trim());
    }

    // Add known missing words back in (ugh)
    let input = File::open(filepath_include).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        let l = l.trim();
        if new_word_frequency.contains_key(l) {
            println!("{} is already found in the dictionary! Skipping!", l);
        } else {
            new_word_frequency.insert(l.to_string(), MINIMUM_FREQUENCY);
        }
    }

    new_word_frequency
}

fn clean_french(
    word_frequency: &HashMap<String, i32>,
    filepath_exclude: &str,
    filepath_include: &str,
) -> HashMap<String, i32> {
    let letters: std::collections::HashSet<char> = "abcdefghijklmnopqrstuvwxyzéàèùâêîôûëïüÿçœæ"
        .chars()
        .collect();

    let mut new_word_frequency = HashMap::new();

    for (key, &value) in word_frequency.iter() {
        let word_chars: std::collections::HashSet<_> = key.chars().collect();

        // Remove words with invalid characters and words that start with a double "a"
        if !word_chars.is_subset(&letters) || key.starts_with("aa") || value <= MINIMUM_FREQUENCY {
            continue;
        }

        new_word_frequency.insert(key.clone(), value);
    }

    // Remove flagged misspellings
    let input = File::open(filepath_exclude).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        new_word_frequency.remove(l.trim());
    }

    // Add known missing words back in (ugh)
    let input = File::open(filepath_include).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        let l = l.trim();
        if new_word_frequency.contains_key(l) {
            println!("{} is already found in the dictionary! Skipping!", l);
        } else {
            new_word_frequency.insert(l.to_string(), MINIMUM_FREQUENCY);
        }
    }

    new_word_frequency
}

fn clean_portuguese(
    word_frequency: &HashMap<String, i32>,
    filepath_exclude: &str,
    filepath_include: &str,
) -> HashMap<String, i32> {
    let letters: std::collections::HashSet<char> =
        "abcdefghijklmnopqrstuvwxyzáâãàçéêíóôõú".chars().collect();

    let mut new_word_frequency = HashMap::new();

    for (key, &value) in word_frequency.iter() {
        let word_chars: std::collections::HashSet<_> = key.chars().collect();

        // Remove words with invalid characters and words that start with a double "a"
        if !word_chars.is_subset(&letters) || key.starts_with("aa") || value <= MINIMUM_FREQUENCY {
            continue;
        }

        new_word_frequency.insert(key.clone(), value);
    }

    // Remove flagged misspellings
    let input = File::open(filepath_exclude).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        new_word_frequency.remove(l.trim());
    }

    // Add known missing words back in (ugh)
    let input = File::open(filepath_include).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        let l = l.trim();
        if new_word_frequency.contains_key(l) {
            println!("{} is already found in the dictionary! Skipping!", l);
        } else {
            new_word_frequency.insert(l.to_string(), MINIMUM_FREQUENCY);
        }
    }

    new_word_frequency
}

fn clean_russian(
    word_frequency: &HashMap<String, i32>,
    filepath_exclude: &str,
    filepath_include: &str,
) -> HashMap<String, i32> {
    let letters: std::collections::HashSet<char> =
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя".chars().collect();
    let vowels: std::collections::HashSet<char> = "аеёиоуыэюя".chars().collect();

    let mut new_word_frequency = HashMap::new();

    for (key, &value) in word_frequency.iter() {
        let word_chars: std::collections::HashSet<_> = key.chars().collect();

        // Remove words with invalid characters, words that start with a double "а" or "ээ",
        // words without a vowel, and words with frequency less than MINIMUM_FREQUENCY
        if !word_chars.is_subset(&letters)
            || (!key.starts_with("аа")
                && key.starts_with("аарон")
                && key.starts_with("аарона")
                && key.starts_with("аарону"))
            || (!key.starts_with("ээ") && key.starts_with("ээг"))
            || key.contains("..")
            || word_chars.is_disjoint(&vowels)
            || value <= MINIMUM_FREQUENCY
        {
            continue;
        }

        new_word_frequency.insert(key.clone(), value);
    }

    // Remove flagged misspellings
    let input = File::open(filepath_exclude).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        new_word_frequency.remove(l.trim());
    }

    // Add known missing words back in (ugh)
    let input = File::open(filepath_include).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        let l = l.trim();
        if new_word_frequency.contains_key(l) {
            println!("{} is already found in the dictionary! Skipping!", l);
        } else {
            new_word_frequency.insert(l.to_string(), MINIMUM_FREQUENCY);
        }
    }

    new_word_frequency
}

fn clean_arabic(
    word_frequency: &HashMap<String, i32>,
    filepath_exclude: &str,
    filepath_include: &str,
) -> HashMap<String, i32> {
    let letters: std::collections::HashSet<char> =
        "دجحإﻹﻷأآﻵخهعغفقثصضذطكمنتالبيسشظزوةىﻻرؤءئ".chars().collect();

    let mut new_word_frequency = HashMap::new();

    for (key, &value) in word_frequency.iter() {
        let word_chars: std::collections::HashSet<_> = key.chars().collect();

        // Remove words with invalid characters, words containing "..",
        // and words with frequency less than MINIMUM_FREQUENCY
        if !word_chars.is_subset(&letters) || key.contains("..") || value <= MINIMUM_FREQUENCY {
            continue;
        }

        new_word_frequency.insert(key.clone(), value);
    }

    // Remove flagged misspellings
    let input = File::open(filepath_exclude).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        new_word_frequency.remove(l.trim());
    }

    // Add known missing words back in (ugh)
    let input = File::open(filepath_include).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        let l = l.trim();
        if new_word_frequency.contains_key(l) {
            println!("{} is already found in the dictionary! Skipping!", l);
        } else {
            new_word_frequency.insert(l.to_string(), MINIMUM_FREQUENCY);
        }
    }

    new_word_frequency
}

fn clean_basque(
    word_frequency: &mut HashMap<String, i32>,
    filepath_exclude: &str,
    filepath_include: &str,
) -> HashMap<String, i32> {
    let letters: std::collections::HashSet<char> = "abcdefghijklmnopqrstuvwxyzñ".chars().collect();

    let mut new_word_frequency = HashMap::new();

    for (key, &value) in word_frequency.iter() {
        let word_chars: std::collections::HashSet<_> = key.chars().collect();

        // Remove words with invalid characters, words starting with "aa",
        // and words with frequency less than MINIMUM_FREQUENCY
        if !word_chars.is_subset(&letters) || key.starts_with("aa") || value <= MINIMUM_FREQUENCY {
            continue;
        }

        new_word_frequency.insert(key.clone(), value);
    }

    // Remove flagged misspellings
    let input = File::open(filepath_exclude).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        new_word_frequency.remove(l.trim());
    }

    // Add known missing words back in
    let input = File::open(filepath_include).unwrap();
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        let l = line.unwrap();
        let l = l.trim();
        if new_word_frequency.contains_key(l) {
            println!("{} is already found in the dictionary! Skipping!", l);
        } else {
            new_word_frequency.insert(l.to_string(), MINIMUM_FREQUENCY);
        }
    }

    new_word_frequency
}

pub fn clean_latvian(
    mut word_frequency: HashMap<String, u32>,
    filepath_exclude: &str,
    filepath_include: &str,
) -> HashMap<String, u32> {
    let letters: HashSet<char> = "aābcčdeēfgģhiījkķlļmnņoprsštuūvzž".chars().collect();
    let vowels: HashSet<char> = "aāiīeēouū".chars().collect();

    word_frequency.retain(|key, _| {
        let key_letters: HashSet<char> = key.chars().collect();
        !key_letters.is_disjoint(&letters)
    });

    word_frequency.retain(|key, _| {
        let key_letters: HashSet<char> = key.chars().collect();
        !key_letters.is_disjoint(&vowels)
    });

    word_frequency.retain(|key, _| !key.contains(".."));

    word_frequency.retain(|key, _| !(key.starts_with("aa") || key.starts_with("ii")));

    word_frequency.retain(|key, _| key.len() > 1);

    word_frequency.retain(|key, _| match word_frequency.get(key) {
        Some(&value) => value > MINIMUM_FREQUENCY,
        None => false,
    });

    let lines_to_remove = lines_from_file(filepath_exclude).expect("Couldn't read the file");
    for line in lines_to_remove {
        word_frequency.remove(&line);
    }

    let lines_to_add = lines_from_file(filepath_include).expect("Couldn't read the file");
    for line in lines_to_add {
        if word_frequency.contains_key(&line) {
            println!("{} is already found in the dictionary! Skipping!", line);
        } else {
            word_frequency.insert(line, MINIMUM_FREQUENCY);
        }
    }

    word_frequency
}

fn lines_from_file(filename: impl AsRef<Path>) -> io::Result<Vec<String>> {
    let file = File::open(filename)?;
    io::BufReader::new(file).lines().collect()
}

fn main() {
    let matches =
        App::new("Build a new dictionary (word frequency) using the OpenSubtitles2018 project")
            .arg(
                Arg::with_name("language")
                    .short("l")
                    .long("language")
                    .required(true)
                    .help("The language being built")
                    .possible_values(&LANGUAGES),
            )
            .arg(
                Arg::with_name("file-path")
                    .short("f")
                    .long("file-path")
                    .help("The path to the downloaded text file OR the saved word frequency json"),
            )
            .arg(
                Arg::with_name("parse-input")
                    .short("p")
                    .long("parse-input")
                    .help("Add this if providing a text file to be parsed")
                    .takes_value(false),
            )
            .arg(
                Arg::with_name("misfit-file")
                    .short("m")
                    .long("misfit-file")
                    .help("Create file with words which was removed from dictionary")
                    .takes_value(false),
            )
            .get_matches();

    // Validate the file path if it was given
    let mut file_path = matches.value_of("file-path").map(Path::new);
    if matches.is_present("parse-input") && file_path.is_none() {
        panic!("A path is required if parsing a text file!");
    }

    if let Some(path) = file_path {
        if !path.exists() {
            panic!("File Not Found. A valid path is required if parsing a text file!");
        }
        file_path = Some(path.canonicalize().unwrap());
    }

    // Get paths for script, module, resources and data directories
    let script_path = env::current_dir().unwrap();
    let module_path = script_path.parent().unwrap().to_path_buf();
    let resources_path = module_path.join("resources");
    let data_path = script_path.join("data");

    println!("{:?}", script_path);
    println!("{:?}", module_path);
    println!("{:?}", resources_path);

    let language = matches.value_of("language").unwrap();

    let exclude_filepath = data_path.join(format!("{}_exclude.txt", language));
    let include_filepath = data_path.join(format!("{}_include.txt", language));

    println!("{:?}", exclude_filepath);
    println!("{:?}", include_filepath);

    // More code to process the data goes here...

    // After the data has been processed, it can be written to a file
    let word_frequency_path = script_path.join(format!("{}.json", language));
    println!("{:?}", word_frequency_path);

    // If the 'misfit-file' flag was given, create that file too
    if matches.is_present("misfit-file") {
        let misfit_filepath = data_path.join(format!("{}_misfit.txt", language));
        println!("{:?}", misfit_filepath);
    }
}
