use headless_chrome::{Browser, LaunchOptionsBuilder};
use rand::Rng;
use std::error::Error;
use std::io::Write;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let options = LaunchOptionsBuilder::default()
        .headless(false)
        .build()
        .unwrap();

    let browser = Browser::new(options).unwrap();
    let url =
        "https://www.google.com/about/careers/applications/jobs/results?q=software%20engineering"
            .to_string();

    //If you want to crawl all ~2.5k jobs
    //let url = "https://www.google.com/about/careers/applications/jobs/results".to_owned();

    let base_query: String = "https://www.google.com/about/careers/applications/".to_string();
    let mut page_id = 0;

    let mut link_vec: Vec<String> = vec![];

    let file = std::fs::File::create("job_links.csv").unwrap();
    let mut writer = std::io::BufWriter::new(file);

    if let Ok(page) = browser.new_tab() {
        loop {
            //let next = format!("{url}?page={page_id}");
            let next = format!("{url}&page={page_id}");
            let url = if page_id == 0 { &url } else { &next };
            let random_delay: u64 = rand::thread_rng().gen_range(80..=280) + 650; //ms

            println!("Opening: {url}\n");

            //wait a bit to load the page
            if let Ok(tab) = page.navigate_to(url) {
                std::thread::sleep(Duration::from_millis(random_delay));

                match tab.find_elements("div .VfPpkd-dgl2Hf-ppHlrf-sM5MNb > div > a") {
                    Ok(elms) => {
                        elms.iter().for_each(|e| {
                            if let Ok(href) = e.get_attribute_value("href") {
                                if let Some(link) = href {
                                    //Push to list
                                    let full_link = format!("{base_query}{link}");
                                    // println!("Found Job link : {full_link}");
                                    link_vec.push(full_link);
                                }
                            }
                        });

                        batch_write_to_csv(&link_vec, &mut writer);
                        writer.flush()?;
                        link_vec.clear();
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
            page_id += 1;
        }
        let _ = page.close_target();
        writer.flush()?;
    }

    Ok(())
}

fn batch_write_to_csv(data: &Vec<String>, writer: &mut std::io::BufWriter<std::fs::File>) {
    data.iter().for_each(
        |link| match writer.write_all(format!("{link},\n").as_bytes()) {
            Err(e) => eprintln!("Csv write Error:  {e}"),
            _ => (),
        },
    )
}
