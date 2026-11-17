import React,{useEffect,useState} from "react";
import {createRoot} from "react-dom/client";
import {ArrowRight,ArrowLeft,Check,CheckCircle2,ChevronDown,FileText,LockKeyhole,ShieldCheck,Sparkles,Wallet,UserRound,House,RotateCcw} from "lucide-react";
import "./style.css";

const API="http://127.0.0.1:8000/api";
const initial={person_age:32,person_gender:"male",person_education:"Bachelor",person_income:60000,person_emp_exp:8,person_home_ownership:"RENT",loan_amnt:10000,loan_int_rate:11,loan_intent:"PERSONAL",cb_person_cred_hist_length:6,credit_score:680,previous_loan_defaults_on_file:"No"};
const money=n=>new Intl.NumberFormat("en-IN",{style:"currency",currency:"INR",maximumFractionDigits:0}).format(Number(n)||0);
async function request(path,options){const r=await fetch(API+path,options);const d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.detail||"We couldn't complete the assessment. Please try again.");return d}
function Field({label,children,hint}){return <label className="field"><span>{label}</span>{children}{hint&&<small>{hint}</small>}</label>}
function App(){
 const [form,setForm]=useState(initial),[step,setStep]=useState(0),[result,setResult]=useState(null),[busy,setBusy]=useState(false),[error,setError]=useState(""),[online,setOnline]=useState(false);
 useEffect(()=>{let active=true;const check=()=>request("/health").then(x=>{if(active)setOnline(Boolean(x.model_available))}).catch(()=>{if(active)setOnline(false)});check();const timer=setInterval(check,3000);return()=>{active=false;clearInterval(timer)}},[]);
 const update=(key,value)=>setForm(p=>({...p,[key]:value}));
 const valid=()=>{
  if(step===0 && (form.person_age<18||form.person_age>=84||form.person_emp_exp>form.person_age-18))return "Please check your age and employment experience.";
  if(step===1 && form.person_income<=0)return "Please enter a valid annual income.";
  if(step===2 && (form.loan_amnt<=0||form.loan_int_rate<=0))return "Please check your loan amount and interest rate.";
  return "";
 };
 const next=()=>{const e=valid();if(e){setError(e);return}setError("");setStep(s=>s+1)};
 async function submit(e){e.preventDefault();const problem=valid();if(problem){setError(problem);return}setBusy(true);setError("");try{setResult(await request("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(form)}))}catch(err){setError(err.message)}finally{setBusy(false)}}
 function reset(){setResult(null);setStep(0);setError("");setForm(initial);window.scrollTo({top:0,behavior:"smooth"})}
 const progress=result?100:(step+1)*33.333;
 return <div className="site">
  <header className="site-header"><a className="wordmark" href="#"><span className="logo-mark">s</span><span>smartloan<span className="wordmark-dot">.</span></span></a><nav className="header-nav"><a href="#how-it-works">How it works</a><a href="#about">About this demo</a></nav><div className="secure-label"><LockKeyhole size={14}/> Local demo</div></header>
  <main>
   <section className="hero"><div className="hero-copy"><div className="eyebrow"><span className="eyebrow-dot"/> A simpler way to explore your options</div><h1>A little closer to<br/><em>your next big step.</em></h1><p className="hero-text">Share a few details and explore an indicative loan assessment. It takes just a couple of minutes.</p><div className="hero-points"><div><ShieldCheck/><span><b>Your details stay local</b><small>This demo does not send an application to a lender.</small></span></div><div><FileText/><span><b>Simple, guided steps</b><small>Just the information needed for this assessment.</small></span></div></div></div>
    <div className="hero-art" aria-hidden="true"><div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/><div className="art-card"><div className="art-card-top"><span>YOUR FINANCIAL JOURNEY</span><span className="art-chip"><Sparkles size={13}/></span></div><div className="art-card-title">Make room<br/>for what's next.</div><div className="art-card-bottom"><div className="art-bars"><i/><i/><i/><i/><i/><i/><i/><i/><i/></div><span>ONE STEP AT A TIME</span></div></div><div className="floating-pill"><CheckCircle2 size={16}/> A clearer picture</div></div>
   </section>
   <section className="application-wrap" id="apply"><div className="application-intro"><div className="eyebrow">LET'S GET STARTED</div><h2>{result?"Your assessment":"Tell us a little about yourself."}</h2><p>{result?"Here's what the demo model estimated from the details you shared.":"Complete the steps below to see your indicative assessment."}</p><div className="mini-trust"><LockKeyhole size={14}/> Your information is used only for this on-device demo flow.</div></div>
    <div className="application-card"><div className="card-topline"><div><span className="card-kicker">{result?"ASSESSMENT COMPLETE":`STEP 0${step+1} OF 03`}</span><h3>{result?"Your result": ["About you","Your finances","Loan details"][step]}</h3></div><span className="step-icon">{result?<CheckCircle2/>:[<UserRound/>,<Wallet/>,<House/>][step]}</span></div><div className="progress-track"><span style={{width:progress+"%"}}/></div>
    {!result?<form onSubmit={step===2?submit:e=>{e.preventDefault();next()}}>
      {step===0&&<div className="fields-grid">
       <Field label="Your age"><input type="number" min="18" max="83" step="1" value={form.person_age} onChange={e=>update("person_age",Number(e.target.value))}/></Field>
       <Field label="Gender"><select value={form.person_gender} onChange={e=>update("person_gender",e.target.value)}><option value="male">Male</option><option value="female">Female</option></select></Field>
       <Field label="Highest education"><select value={form.person_education} onChange={e=>update("person_education",e.target.value)}>{["High School","Associate","Bachelor","Master","Doctorate"].map(x=><option key={x}>{x}</option>)}</select></Field>
       <Field label="Years of work experience"><input type="number" min="0" max="65" step="1" value={form.person_emp_exp} onChange={e=>update("person_emp_exp",Number(e.target.value))}/></Field>
      </div>}
      {step===1&&<div className="fields-grid">
       <Field label="Annual income" hint="Enter your yearly income in Indian rupees."><div className="input-prefix"><span>₹</span><input type="number" min="1" step="1" value={form.person_income} onChange={e=>update("person_income",Number(e.target.value))}/></div></Field>
       <Field label="Current housing situation"><select value={form.person_home_ownership} onChange={e=>update("person_home_ownership",e.target.value)}><option value="RENT">Renting</option><option value="OWN">Own my home</option><option value="MORTGAGE">Paying a mortgage</option><option value="OTHER">Other</option></select></Field>
       <div className="field full-field"><span>Annual income entered</span><div className="income-preview">{money(form.person_income)} <small>per year</small></div></div>
      </div>}
      {step===2&&<div className="fields-grid">
       <Field label="Loan amount requested" hint="How much would you like to borrow?"><div className="input-prefix"><span>₹</span><input type="number" min="1" step="1" value={form.loan_amnt} onChange={e=>update("loan_amnt",Number(e.target.value))}/></div></Field>
       <Field label="Loan purpose"><select value={form.loan_intent} onChange={e=>update("loan_intent",e.target.value)}><option value="PERSONAL">Personal expenses</option><option value="EDUCATION">Education</option><option value="MEDICAL">Medical expenses</option><option value="VENTURE">Business</option><option value="HOMEIMPROVEMENT">Home improvement</option><option value="DEBTCONSOLIDATION">Debt consolidation</option></select></Field>
       <Field label="Credit history length (years)"><input type="number" min="0" max="70" step="1" value={form.cb_person_cred_hist_length} onChange={e=>update("cb_person_cred_hist_length",Number(e.target.value))}/></Field>
       <Field label="Credit score"><input type="number" min="300" max="900" step="1" value={form.credit_score} onChange={e=>update("credit_score",Number(e.target.value))}/></Field>
       <Field label="Interest rate (%)"><input type="number" min=".01" max="100" step="any" value={form.loan_int_rate} onChange={e=>update("loan_int_rate",Number(e.target.value))}/></Field>
       <Field label="Previous loan default on record"><select value={form.previous_loan_defaults_on_file} onChange={e=>update("previous_loan_defaults_on_file",e.target.value)}><option value="No">No</option><option value="Yes">Yes</option></select></Field>
       <div className="summary-strip full-field"><span>Loan amount</span><b>{money(form.loan_amnt)}</b><span>Annual income</span><b>{money(form.person_income)}</b></div>
      </div>}
      {error&&<div className="form-error">{error}</div>}
      <div className="form-actions">{step>0&&<button type="button" className="back-button" onClick={()=>{setError("");setStep(s=>s-1)}}><ArrowLeft size={16}/> Back</button>}<button className="primary-button" type="submit" disabled={busy||!online}>{busy?"Preparing your assessment…":step===2?"Get my assessment":"Continue"}{!busy&&<ArrowRight size={17}/>}</button></div>
      {!online&&<p className="offline-note">Assessment service is starting. Please wait a moment and refresh if this message remains.</p>}
     </form>:<div className="result-panel"><div className="result-symbol"><CheckCircle2 size={28}/></div><div className="result-overline">YOUR INDICATIVE RESULT</div><h4>{predictionLabel(result)}</h4><p className="result-description">The model's estimated likelihood of a positive outcome in its reference dataset is shown below.</p><div className="score-block"><div><span>Positive dataset-label likelihood</span><b>{(Number(result.class_1_probability)*100).toFixed(1)}%</b></div><div className="score-track"><span style={{width:(Number(result.class_1_probability)*100)+"%"}}/></div></div>{result.input_checks?.warnings?.length>0&&<div className="result-disclaimer"><ShieldCheck size={17}/><p><b>Input checks:</b> {result.input_checks.warnings.join(" ")}</p></div>}{result.dataset_context&&<div className="dataset-context"><b>What the reference data says</b>{Object.entries(result.dataset_context).map(([key,value])=><div key={key}><span>{({person_home_ownership:"Home ownership",loan_intent:"Loan purpose",previous_loan_defaults_on_file:"Previous defaults"})[key]||key}</span><strong>{value.value}</strong><small>{(value.observed_class_1_rate*100).toFixed(1)}% positive labels across {value.sample_count.toLocaleString("en-IN")} similar-category records</small></div>)}</div>}<div className="result-disclaimer"><ShieldCheck size={17}/><p><b>Important:</b> {result.disclaimer} No application has been submitted to a lender.</p></div><div className="form-actions"><button className="back-button" onClick={()=>{setResult(null);setStep(2)}}><ArrowLeft size={16}/> Review details</button><button className="primary-button" onClick={reset}>Start again <RotateCcw size={16}/></button></div></div>}
    </div>
   </section>
   <section className="how-section" id="how-it-works"><div className="section-heading"><div className="eyebrow">A STRAIGHTFORWARD PROCESS</div><h2>Three steps. One clearer picture.</h2><p>No complicated forms. Just a guided demo assessment.</p></div><div className="how-grid"><div className="how-card"><span>01</span><UserRound/><h3>Tell us about you</h3><p>Share a few basic details about your background and finances.</p></div><div className="how-card"><span>02</span><Wallet/><h3>Share your loan details</h3><p>Tell us the amount you're exploring and what it's for.</p></div><div className="how-card"><span>03</span><Sparkles/><h3>See your estimate</h3><p>Get an indicative result from our research model, instantly.</p></div></div></section>
   <section className="about-section" id="about"><div><div className="eyebrow">TRANSPARENCY MATTERS</div><h2>About this demonstration</h2><p>SmartLoan is currently an academic machine-learning prototype. It explores patterns in a synthetic loan-status dataset. It is not connected to a bank, credit bureau, or lending provider. Your entries are processed by the local application and are not submitted as a real loan application.</p></div><div className="about-badge"><ShieldCheck/><span><b>Demo environment</b><small>No real loan applications or credit checks</small></span></div></section>
  </main><footer className="site-footer"><a className="wordmark" href="#"><span className="logo-mark">s</span><span>smartloan<span className="wordmark-dot">.</span></span></a><span>© 2026 SmartLoan Research Lab · Academic demonstration</span><a href="#about">Privacy & demo information</a></footer>
 </div>
}
function predictionLabel(r){return r.decision==="CLASS_1_MORE_LIKELY"?"Positive dataset outcome more likely":"Negative dataset outcome more likely"}
createRoot(document.getElementById("root")).render(<App/>);
