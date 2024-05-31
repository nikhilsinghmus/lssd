(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[405],{8312:function(e,i,n){(window.__NEXT_P=window.__NEXT_P||[]).push(["/",function(){return n(236)}])},236:function(e,i,n){"use strict";n.r(i),n.d(i,{__N_SSG:function(){return w},default:function(){return y}});var t=n(5893);n(7294);var r=n(2445),a=n(9236),l=n(5117),s=n(4685),o=n(6137),c=n(5328),d=n(3523),h=n(2489),u=n(9417),f=n(9814),p=n(825),g=n(4577),x=n(1752),m=n.n(x);let v=e=>{var i;let{examples:n}=e,s=null===(i=m()().publicRuntimeConfig)||void 0===i?void 0:i.basePath;n.map(e=>e.pkg_id).filter((e,i,n)=>n.indexOf(e)===i);let o=n.map(e=>e.lang_id).filter((e,i,n)=>n.indexOf(e)===i),c=n.map(e=>e.seq_id).filter((e,i,n)=>n.indexOf(e)===i);return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(a.D,{order:3,align:"center",pt:"xl",children:"Example Clips"}),c.map((e,i)=>(0,t.jsxs)(r.W,{fluid:!0,pt:"xl",pb:"xl",children:[(0,t.jsx)(a.D,{order:4,align:"center",children:"Clip "+(i+1)}),(0,t.jsx)(f.M,{cols:o.length,children:n.filter(i=>i.seq_id===e).map((e,i)=>(0,t.jsxs)(p.Z,{style:{width:"100%",height:"auto"},children:[(0,t.jsx)(p.Z.Section,{children:(0,t.jsx)(l.x,{align:"center",children:e.lang_id.split(".")[0].toUpperCase()})}),(0,t.jsx)(p.Z.Section,{children:(0,t.jsx)(g.o,{ratio:1,children:(0,t.jsx)("video",{controls:!0,preload:"metadata",children:(0,t.jsx)("source",{src:(null!=s?s:"")+"/examples/"+e.video_file+"#t=0.001",type:"video/mp4"})})})})]},i))})]},i))]})};var j={title:"Looking Similar, Sounding Different",subtitle:"Leveraging Counterfactual Cross-Modal Pairs for Audiovisual Representation Learning",authors:[{name:"Nikhil Singh",affiliation:"MIT",url:"http://media.mit.edu/~nsingh1",email:"nsingh1@mit.edu"},{name:"Chih-Wei Wu",affiliation:"Netflix",url:"https://scholar.google.com/citations?user=n8eQV6kAAAAJ&hl=en&oi=ao",email:"chihweiw@netflix.com"},{name:"Iroro Orife",affiliation:"Netflix",url:"https://scholar.google.com/citations?user=h1KyIt0AAAAJ&hl=en&oi=ao",email:"iorife@netflix.com"},{name:"Mahdi M. Kalayeh",affiliation:"Netflix",url:"https://scholar.google.com/citations?user=gleejrUAAAAJ&hl=en&oi=ao",email:"mkalayeh@netflix.com"}],paperURL:"https://arxiv.org/abs/2304.05600",codeURL:"",venue:"CVPR 2024",abstract:"Audiovisual representation learning typically relies on the correspondence between sight and sound. However, there are often multiple audio tracks that can correspond with a visual scene. Consider, for example, different conversations on the same crowded street. The effect of such counterfactual pairs on audiovisual representation learning has not been previously explored. To investigate this, we use dubbed versions of movies and television shows to augment cross-modal contrastive learning. Our approach learns to represent alternate audio tracks, differing only in speech, similarly to the same video. Our results, from a comprehensive set of experiments investigating different training strategies, show this general approach improves performance on a range of downstream auditory and audiovisual tasks, without majorly affecting linguistic task performance overall. These findings highlight the importance of considering speech variation when learning scene-level audiovisual correspondences and suggest that dubbed audio can be a useful augmentation technique for training audiovisual models toward more robust performance on diverse downstream tasks."};let b=e=>{let{data:i}=e,n=j.authors.map(e=>e.note).filter(e=>void 0!==e).filter((e,i,n)=>n.indexOf(e)===i).map(e=>e),f=n.reduce((e,i,n)=>(e[i]="*".repeat(n+1),e),{}),p=j.authors.map(e=>e.affiliation).filter(e=>void 0!==e).filter((e,i,n)=>n.indexOf(e)===i).map(e=>e),g=n.reduce((e,i,n)=>(e[i]=n+1,e),{});return(0,t.jsxs)(r.W,{fluid:!0,p:"xl",children:[(0,t.jsx)(a.D,{align:"center",children:j.title}),(0,t.jsx)(a.D,{align:"center",order:2,children:j.subtitle}),(0,t.jsx)(l.x,{size:"lg",align:"center",pt:"xl",children:j.authors.map((e,i)=>(0,t.jsxs)("span",{children:[(0,t.jsx)(s.e,{href:e.url,target:"_blank",display:"inline",color:"cyan",children:e.name}),p.length>1?(0,t.jsx)("sup",{children:g[e.affiliation]}):"",e.note?(0,t.jsx)("sup",{children:f[e.note]}):"",i<j.authors.length-1&&", "]},i))}),(0,t.jsx)(l.x,{size:"lg",align:"center",children:p.map((e,i)=>(0,t.jsxs)("span",{children:[p.length>1?(0,t.jsx)("sup",{children:g[e]}):"",e,i<p.length-1&&", "]},i))}),Object.entries(f).map((e,i)=>{let[n,r]=e;return(0,t.jsxs)(l.x,{size:"sm",color:"gray",align:"center",children:[r," ",n]},i)}),(0,t.jsx)(l.x,{size:"xl",align:"center",color:"gray",children:j.venue}),(0,t.jsxs)(o.Z,{position:"center",mt:"xl",mb:"xl",children:[(0,t.jsx)(s.e,{href:j.paperURL,target:"_blank",color:"cyan",size:"xl",weight:"bold",align:"center",children:(0,t.jsx)(c.z,{color:"cyan",leftIcon:(0,t.jsx)(h.G,{icon:u.gMD,size:"lg",style:{verticalAlign:"bottom"}}),children:"Paper"})}),(0,t.jsx)(s.e,{href:j.codeURL,target:"_blank",color:"cyan",size:"xl",weight:"bold",align:"center",children:(0,t.jsx)(c.z,{color:"cyan",leftIcon:(0,t.jsx)(h.G,{icon:u.dT$,size:"lg",style:{verticalAlign:"bottom"}}),children:"Code (Coming Soon)"})})]}),(0,t.jsx)(a.D,{order:4,align:"center",pt:"xl",style:{fontVariant:"small-caps"},children:"Abstract"}),(0,t.jsx)(d.M,{children:(0,t.jsx)(l.x,{maw:1200,align:"justify",children:j.abstract})}),(0,t.jsx)(v,{examples:i})]})};var w=!0,y=b}},function(e){e.O(0,[976,444,774,888,179],function(){return e(e.s=8312)}),_N_E=e.O()}]);