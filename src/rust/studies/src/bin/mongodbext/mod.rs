use mongodb::bson::doc;
use mongodb::Client;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub struct Creds_lev0 {
    user: String,
    passwd: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Creds {
    TS_DB: Creds_lev0,
}

#[derive(Debug)]
pub struct Credentials {
    pub credentials: Creds,
}

impl Credentials {

    pub async fn get_credentials(uri: &str) -> Result<(String,String), Box<dyn std::error::Error>> {
        let client = Client::with_uri_str(uri).await?;
        let database = client.database("MULTISCALEGB");
        let collection = database.collection::<Creds>("credentials");
        let creds = Credentials{credentials: collection.find_one(None, None).await?.unwrap()};

        let user = creds.get_user();
        let passwd = creds.get_passwd();
        Ok((user, passwd))
    }

    pub async fn get_credentials_static(uri: &str) -> Result<(String,String), Box<dyn std::error::Error>> {
        let user: String = String::from("charles");
        let passwd: String = String::from("gongzuo");
        Ok((user, passwd))
    }

    fn new(user: String, passwd: String) -> Self {
        let creds_lev0 = Creds_lev0{user, passwd};
        let creds = Creds{TS_DB: creds_lev0};
        Credentials{credentials: creds}
    }

    pub fn get_user(&self) -> String {
        self.credentials.TS_DB.user.clone()
    }

    pub fn get_passwd(&self) -> String {
        self.credentials.TS_DB.passwd.clone()
    }
}

