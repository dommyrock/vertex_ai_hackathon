use headless_chrome::{Browser, LaunchOptionsBuilder};
use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use tokio::time::Duration;

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct Job {
    url: String,
    body: String,
    salary: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let options = LaunchOptionsBuilder::default()
        .headless(false)
        .build()
        .unwrap();

    let browser = Browser::new(options).unwrap();

    let jobs = include_str!("../job_links.csv")
        .split_terminator(',')
        .collect::<Vec<&str>>();

    let mut job_idx = 0;
    if let Ok(page) = browser.new_tab() {
        loop {
            let url = jobs[job_idx];
            //https://www.google.com/about/careers/applications/jobs/results/119975360088416966-
            let job_id: &str = url
                .split("results/")
                .nth(1)
                .unwrap()
                .split("-")
                .next()
                .unwrap();
            println!("Job ID: {job_id}\n");
            let random_delay: u64 = rand::thread_rng().gen_range(80..=280) + 650; //ms
            if let Ok(tab) = page.navigate_to(url) {
                std::thread::sleep(Duration::from_millis(random_delay)); //wait 1s to load the page

                //Remove as much noise as possible (for tokenization & embeddings steps)
                match tab.find_element("main > div .DkhPwc") {
                    Ok(elm) => {
                        let file_name = format!("{job_idx}_jobId_{job_id}.txt");
                        let path = format!("documents/{file_name}");
                        let file = File::create(path)?;
                        let mut writer = std::io::BufWriter::new(file);

                        //Title
                        if let Ok(title) = elm.find_element("div .sPeqm > h2") {
                            if let Ok(txt) = title.get_inner_text() {
                                let t = format!("Job title:\n{txt}\n");
                                // println!("{t}");
                                writer.write(t.as_bytes())?;
                            }
                        }
                        //Location
                        if let Ok(elm) = elm.find_element(".pwO9Dc") {
                            if let Ok(txt) = elm.get_inner_text() {
                                let ln = txt
                                    .lines()
                                    .filter_map(|ln| {
                                        (!ln.starts_with("place") && !ln.starts_with("; +"))
                                            .then_some(ln)
                                    })
                                    .map(|f| f.trim_start_matches("; "))
                                    .collect::<Vec<&str>>();
                                let t= format!("\nLocation:");
                                let t2 = format!("\n{ln:?}\n");
                                // print!("{t}");
                                // print!("{t2}");
                                writer.write(t.as_bytes())?;
                                writer.write(t2.as_bytes())?;
                            }
                        }
                        //All stats
                        //"corporate_fare\nGoogle\nplace\nMunich, Germany\n; Berlin, Germany\n; +1 more\nbar_chart\nMid"
                        if let Ok(span) = elm.find_element(".op1BBf") {
                            if let Ok(txt) = span.get_inner_text() {
                                //Push to list
                                let stats: Vec<&str> = txt
                                    .split("\n")
                                    .filter_map(|s| {
                                        (!s.starts_with("corporate")
                                            && !s.starts_with("; +")
                                            && !s.starts_with("bar_chart"))
                                        .then_some(s)
                                    })
                                    .map(|f| f.trim_start_matches("; "))
                                    .collect();
                                if let Some(workplace) = stats.first() {
                                    let t = format!("\nWorkplace: {workplace}\n");
                                    // print!("{t}");
                                    writer.write(t.as_bytes())?;
                                }
                                //Location + Seniority stats
                                let mut stats = stats.into_iter().skip(2).collect::<Vec<&str>>();
                                let job_lvl = stats.remove(stats.len() - 1);

                                let t = format!("\nAll locations: {stats:?}");
                                let t2 = format!("\nSeniority: {job_lvl}\n");
                                // print!("{t}");
                                // print!("{t2}");
                                writer.write(t.as_bytes())?;
                                writer.write(t2.as_bytes())?;
                            }
                        }
                        //NOTE / usually contains preferred working location info (only present if there are more locations)
                        if let Ok(span) = elm.find_element(".MyVLbf") {
                            if let Ok(txt) = span.get_inner_text() {
                                let t = format!("\nNOTE and Preferred working location:");
                                let t2 = format!("\n{txt}\n");
                                // print!("{t}");
                                // print!("{t2}");
                                writer.write(t.as_bytes())?;
                                writer.write(t2.as_bytes())?;
                            }
                        }
                        //Qualifiacations
                        if let Ok(ul) = elm.find_elements("div .KwJkGe > ul") {
                            for (i, u) in ul.iter().enumerate() {
                                if let Ok(txt) = u.get_inner_text() {
                                    if i == 0 {
                                        let t = format!("\nMin Qualifications: \n{txt}");
                                        // print!("{t}");
                                        writer.write(t.as_bytes())?;
                                    } else {
                                        let t = format!("\nPreferred Qualifications: \n{txt}");
                                        // print!("{t}");
                                        writer.write(t.as_bytes())?;
                                    }
                                }
                            }
                        }
                        //About the job
                        if let Ok(div) = elm.find_element("div .aG5W3") {
                            if let Ok(txt) = div.get_inner_text() {
                                let t = format!("\n{txt}\n");
                                // print!("{t}");
                                writer.write(t.as_bytes())?;
                            }
                        }
                        //Responsibilities
                        if let Ok(div) = elm.find_element("div .BDNOWe") {
                            if let Ok(txt) = div.get_inner_text() {
                                let t = format!("\n{txt}\n");
                                // print!("{t}");
                                writer.write(t.as_bytes())?;
                            }
                        }

                        writer.flush().expect(&format!(
                            "Failed to flush the file buffer for {}",
                            file_name
                        ));
                    }
                    Err(e) => {
                        println!("\nError finding element on {}\nERR: {}\n", &url, e);
                        break;
                    }
                }
            } else {
                println!("Navigation to {url} failed ");
                break;
            }
            job_idx += 1;
        }
        let _ = page.close_target();
    }

    Ok(())
}
