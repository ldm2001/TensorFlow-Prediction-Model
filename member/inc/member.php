<?php
// Member Class file
class Member {
  // 멤버 변수, 프로퍼티
  private $conn;

  // 생성자
  public function __construct($db) {
    $this->conn = $db;
  }

  // 아이디 중복체크용 멤버 함수, 메소드
  public function id_exists($id) {
    $sql = "SELECT * FROM member WHERE id=:id";
    $stmt = $this->conn->prepare($sql);
    $stmt->bindParam(':id', $id);
    $stmt->execute();

    return $stmt->rowCount() ? true : false;
  }

  // 이메일 형식 체크
  public function email_format_check($email) {
    return filter_var($email, FILTER_VALIDATE_EMAIL);
  }

  public function email_exists($email) {
    $sql = "SELECT * FROM member WHERE email=:email";
    $stmt = $this->conn->prepare($sql);
    $stmt->bindParam(':email', $email);
    $stmt->execute();

    return $stmt->rowCount() ? true : false;    
  }

  // 회원정보 입력
  public function input($marr) {

    // 단방향 암호화
    $new_hash_password = password_hash($marr['password'], PASSWORD_DEFAULT);

    $sql = "INSERT INTO member(id, name, password, email, zipcode, addr1, addr2, photo, create_at, ip) VALUES
            (:id, :name, :password, :email, :zipcode, :addr1, :addr2, :photo, NOW(), :ip)";
    $stmt = $this->conn->prepare($sql);
    $stmt->bindParam(':email'   , $marr['email']);
    $stmt->bindParam(':id'      , $marr['id']);
    $stmt->bindParam(':name'    , $marr['name']);
    $stmt->bindParam(':password', $new_hash_password);
    $stmt->bindParam(':email'   , $marr['email']);
    $stmt->bindParam(':zipcode' , $marr['zipcode']);
    $stmt->bindParam(':addr1'   , $marr['addr1']);
    $stmt->bindParam(':addr2'   , $marr['addr2']);
    $stmt->bindParam(':photo'   , $marr['photo']);
    $stmt->bindParam(':ip'      , $_SERVER['REMOTE_ADDR']);
    $stmt->execute();
  }

  // 로그인
  public function login($id, $pw) {

    // password_verify($password, $new_password)

    $sql = "SELECT password FROM member WHERE id=:id";
    $stmt = $this->conn->prepare($sql);
    $stmt->bindParam(':id', $id);

    $stmt->execute();

    if( $stmt->rowCount() ) {
      $row = $stmt->fetch();

      if ( password_verify($pw, $row['password']) ) {
        $sql = "UPDATE member SET login_dt=NOW() WHERE id=:id"  ;
        $stmt = $this->conn->prepare($sql); 
        $stmt->bindParam(':id', $id);
        $stmt->execute();
    
        return true;
      } else {
        return false;
      }

    } else {
      return false;
    }
  }

  public function logout() {
    session_start();
    session_destroy();

    die('<script>self.location.href="../index.php";</script>');
  }

  public function getInfoFormIdx($idx) {
    $sql = "SELECT * FROM member WHERE idx=:idx";
    $stmt = $this->conn->prepare($sql);
    $stmt->bindParam(":idx", $idx);
    $stmt->setFetchMode(PDO::FETCH_ASSOC);
    $stmt->execute();
    return $stmt->fetch();
  }

  public function getInfo($id) {
    $sql = "SELECT * FROM member WHERE id=:id";
    $stmt = $this->conn->prepare($sql);
    $stmt->bindParam(":id", $id);
    $stmt->setFetchMode(PDO::FETCH_ASSOC);
    $stmt->execute();
    return $stmt->fetch();
  }

  public function edit($marr) {
    $sql = "UPDATE member SET name=:name, email=:email, zipcode=:zipcode, addr1=:addr1, addr2=:addr2, photo=:photo ";
    $params = [
      ':name' => $marr['name'], 
      ':email' => $marr['email'],
      ':zipcode' => $marr['zipcode'],
      ':addr1' => $marr['addr1'],
      ':addr2' => $marr['addr2'],
      ':photo' => $marr['photo']
    ];

    if($marr['password'] != '') {
      
      // 단방향 암호화
      $new_hash_password = password_hash($marr['password'], PASSWORD_DEFAULT);
      $params[':password'] = $new_hash_password;

      $sql .= ", password=:password";
    }

    if($_SESSION['ses_level'] == 10 && isset($marr['idx']) && $marr['idx'] != '') {
      $params[':level'] = $marr['level'];
      $params[':idx'] = $marr['idx'];
      $sql .= ", level=:level";
      $sql .= " WHERE idx=:idx";  

    } else {
      $params[':id'] = $marr['id'];
      $sql .= " WHERE id=:id";
    }

    $stmt = $this->conn->prepare($sql);
    $stmt->execute($params);
  }

  // 회원목록
  public function list($page, $limit, $paramArr) {
    $start = ($page - 1) * $limit; 

    $where = "";
    if($paramArr['sn'] != '' && $paramArr['sf'] != '') {
      switch($paramArr['sn']) {
        case 1 : $sn_str = 'name'; break;
        case 2 : $sn_str = 'id'; break;
        case 3 : $sn_str = 'email'; break;
      }

      $where = " WHERE ". $sn_str."=:sf ";
    }

    $sql = "SELECT idx, id, name, email, DATE_FORMAT(create_at,'%Y-%m-%d %H:%i') AS create_at 
    FROM member ". $where ."
    ORDER BY idx DESC LIMIT ".$start.",". $limit; 

    $stmt = $this->conn->prepare($sql);

    if($where != '') {
      $stmt->bindParam(':sf', $paramArr['sf']);
    }

    $stmt->setFetchMode(PDO::FETCH_ASSOC);
    $stmt->execute();
    return $stmt->fetchAll();
  }

  public function total($paramArr) {

    $where = "";
    if($paramArr['sn'] != '' && $paramArr['sf'] != '') {
      switch($paramArr['sn']) {
        case 1 : $sn_str = 'name'; break;
        case 2 : $sn_str = 'id'; break;
        case 3 : $sn_str = 'email'; break;
      }

      $where = " WHERE ". $sn_str."=:sf ";
    }

    $sql = "SELECT COUNT(*) cnt FROM member ". $where;
    $stmt = $this->conn->prepare($sql);
    if($where != '') {
      $stmt->bindParam(':sf', $paramArr['sf']);
    }

    $stmt->setFetchMode(PDO::FETCH_ASSOC);
    $stmt->execute();
    $row = $stmt->fetch();    
    return $row['cnt'];
  }

  // 회원목록
  public function getAllData() {

    $sql = "SELECT * FROM member ORDER BY idx ASC";
    $stmt = $this->conn->prepare($sql);
    $stmt->setFetchMode(PDO::FETCH_ASSOC);
    $stmt->execute();
    return $stmt->fetchAll();
  }

  // 회원삭제
  public function member_del($idx) {
    
    $sql = "DELETE FROM member WHERE idx=:idx";
    $stmt = $this->conn->prepare($sql);
    $stmt->bindParam(':idx', $idx);
    $stmt->execute();
  }

  // 프로필 이미지 업로드
  public function profile_upload($id, $new_photo, $old_photo = '') {
    if($old_photo != '') {
      unlink(PROFILE_DIR. $old_photo); 
    }
  
    $tmparr = explode('.', $new_photo['name']); 
    $ext = end($tmparr); 
    $photo = $id .'.'. $ext; // 새로운 파일명 
  
    copy($new_photo['tmp_name'], PROFILE_DIR."/". $photo);
  
    return $photo;
  }

}